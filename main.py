#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from datetime import timedelta

import requests
from auth import authenticate_user, create_access_token, get_current_user
from config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JOBQUEUE_MAX_JOBS,
    JOBQUEUE_PER_JOB_CONCURRENCY,
    RATE,
    logger,
)
from db import get_db
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jobs import JobQueue, JobStatusEnum
from models import MultiTrafficRequest, Token
from models_db import Job, TrafficLog
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from step2_traffic_analysis import GoogleMapsTrafficAnalyzer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from utils import get_job_record, lifespan, update_job

# FastAPI app
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Google Maps Traffic Analyzer API", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429, content={"detail": "Too many requests"}
    ),
)

# static directory
os.makedirs("static/images/traffic_screenshots", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Worker function for a single location
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(2),
    retry=retry_if_exception_type(Exception),
)
def run_single_location_blocking(
    lat,
    lng,
    storefront_direction,
    day_of_week,
    target_time,
    proxy=None,
):
    selenium_url = os.getenv("SELENIUM_URL", "http://selenium-hub:4444/wd/hub")
    analyzer = GoogleMapsTrafficAnalyzer(selenium_url=selenium_url, proxy=proxy)
    return analyzer.analyze_location_traffic(
        lat=lat,
        lng=lng,
        save_to_static=True,
        storefront_direction=storefront_direction,
        day_of_week=day_of_week,
        target_time=target_time,
    )


job_queue = JobQueue(
    worker_callable=run_single_location_blocking,
    max_workers=JOBQUEUE_MAX_JOBS,
    per_job_concurrency=JOBQUEUE_PER_JOB_CONCURRENCY,
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the error details
    logger.error(f"Global error: {str(exc)}")

    # Return a generic error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Something went wrong. Please try again later.",
        },
    )


@app.post("/login", response_model=Token)
@app.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/analyze-traffic")
@app.post("/analyze-batch")
@app.post("/analyze-locations")
@app.post("/analyze-points")
@limiter.limit(RATE)
async def analyze_batch(
    request: Request,
    payload: MultiTrafficRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit a batch (up to 20 locations). Returns job_id immediately.
    Client must poll /job/{job_uid} to get status or results.
    """
    if not payload.locations:
        raise HTTPException(status_code=400, detail="No locations provided")
    if len(payload.locations) > 20:
        raise HTTPException(status_code=400, detail="Max 20 locations per request")

    job_payload = {
        "locations": [
            {
                "lat": loc.lat,
                "lng": loc.lng,
                "storefront_direction": loc.storefront_direction or "north",
                "day": loc.day,
                "time": loc.time,
            }
            for loc in payload.locations
        ],
        "proxy": payload.proxy,
        "request_base_url": str(request.base_url),
    }

    job_uid = job_queue.submit(job_payload)
    status = JobStatusEnum.PENDING.value

    try:
        job = Job(uuid=job_uid, status=status, user_id=user.id)
        db.add(job)
        db.commit()
    except Exception as e:
        logger.warning(f"DB log failed to create job {job_uid}: {e}")

    return {"job_id": job_uid, "status": status}


@app.get("/job/{job_uid}")
async def get_job(
    job_uid: str | None, user=Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Return the job status if still running,
    or full result when job is completed.
    """

    job = job_queue.get(job_uid)
    if not job:
        result = get_job_record(db, job_uid, user.id)
        if result:
            return result
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")
    remaining = job.get("remaining", 0)
    response = {"job_id": job_uid, "status": status.value}

    if status in (JobStatusEnum.PENDING, JobStatusEnum.RUNNING):
        response["remaining"] = remaining
        return response

    if status == JobStatusEnum.FAILED:
        job_queue.remove(job_uid)

        error = job.get("error")
        response["error"] = error
        update_job(
            db, job_uid, user.id, status=status.value, remaining=remaining, error=error
        )

        raise HTTPException(
            status_code=500,
            detail={"message": "Job execution failed", "details": response},
        )

    # DONE: log results into DB if not already logged
    if status in (JobStatusEnum.DONE, JobStatusEnum.CANCELED) and job.get("result"):
        if not job.get("_logged_to_db"):
            try:
                job_id = db.query(Job).filter(Job.uuid == job_uid).first().id

                results = job["result"].get("results", [])
                payload_locations = job["payload"].get("locations", [])
                for i, res in enumerate(results):
                    log = TrafficLog(
                        lat=payload_locations[i].get("lat"),
                        lng=payload_locations[i].get("lng"),
                        score=res.get("score"),
                        method=res.get("method"),
                        screenshot_url=res.get("screenshot_url"),
                        details=res,
                        job_id=job_id,
                    )
                    db.add(log)
                db.commit()
                job["_logged_to_db"] = True
            except Exception as e:
                logger.warning(f"DB log failed for job {job_uid}: {e}")

    job_queue.remove(job_uid)

    response["remaining"] = remaining
    response["result"] = job.get("result")
    response["error"] = job.get("error")

    update_job(
        db,
        job_uid,
        user.id,
        status=status.value,
        remaining=remaining,
        error=job.get("error"),
    )

    return response


@app.post("/job/{job_uid}/cancel")
async def cancel_job(job_uid: str, user=Depends(get_current_user), db=Depends(get_db)):
    job = job_queue.cancel(job_uid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status").value
    remaining = job.get("remaining", 0)
    error = job.get("error")

    update_job(db, job_uid, user.id, status=status, remaining=remaining, error=error)

    return {
        "job_id": job_uid,
        "status": status,
        "remaining": remaining,
        "result": job.get("result"),  # partial results included
        "error": error,
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify Selenium Grid status
    """
    selenium_status = "unknown"
    grid_capacity = 0
    available_sessions = 0

    try:
        # Check Selenium Grid status
        response = requests.get("http://selenium-hub:4444/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            selenium_status = "healthy"
            grid_capacity = data.get("value", {}).get("ready", False)
            available_sessions = (
                len(data.get("value", {}).get("nodes", [])) * 4
            )  # 4 sessions per node
    except Exception as e:
        selenium_status = f"unhealthy: {str(e)}"

    return {
        "api": "healthy",
        "selenium_grid": selenium_status,
        "concurrent_capacity": grid_capacity,
        "available_sessions": available_sessions,
        "grid_max_sessions": 20,
        "nodes_configured": 5,
        "sessions_per_node": 4,
    }
