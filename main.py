#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from datetime import timedelta

import httpx
from async_worker import job_queue, lifespan
from auth import authenticate_user, create_access_token, get_current_user
from config import ACCESS_TOKEN_EXPIRE_MINUTES, RATE, logger
from db import get_db
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jobs import JobStatusEnum
from models import MultiTrafficRequest, Token, TrafficResponse
from models_db import Job, TrafficLog
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from utils import get_job_record, update_job

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
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(form_data.username, form_data.password, db)
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
    db: AsyncSession = Depends(get_db),
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
        "count": len(payload.locations),
        "proxy": payload.proxy,
        "request_base_url": str(request.base_url),
    }

    job_uid = await job_queue.submit(job_payload)
    status = JobStatusEnum.PENDING.value

    try:
        job = Job(uuid=job_uid, status=status, user_id=user.id)
        db.add(job)
        await db.commit()
    except Exception as e:
        logger.warning(f"DB log failed to create job {job_uid}: {e}")

    return {
        "job_id": job_uid,
        "status": status,
        "locations_count": len(payload.locations),
    }


@app.get("/job/{job_uid}", response_model=TrafficResponse)
async def get_job(
    job_uid: str | None,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the job status if still running, or full result when job is completed.
    """

    job = await job_queue.get(job_uid)
    if not job:
        result = await get_job_record(db, job_uid, user.id)
        if result:
            return result
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")
    response = TrafficResponse(
        job_id=job_uid,
        status=status.value,
        completed=job.get("completed", 0),
        locations_count=job.get("payload", {}).get("count", 0),
        result=job.get("result"),
        error=job.get("error"),
    )

    if status in (JobStatusEnum.PENDING, JobStatusEnum.RUNNING):
        return response

    if status == JobStatusEnum.FAILED:
        await job_queue.remove(job_uid)
        await update_job(
            db,
            job_uid,
            user.id,
            status=response.status,
            error=response.error,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Job execution failed",
                "details": response,
            },
        )

    # DONE: log results to DB
    if status in (JobStatusEnum.DONE, JobStatusEnum.CANCELED) and job.get("result"):
        if not job.get("_logged_to_db"):
            try:
                result = await db.execute(select(Job).filter(Job.uuid == job_uid))
                job_record = result.scalar_one_or_none()

                if job_record:
                    results = job["result"].get("locations", [])
                    payload_locations = job["payload"].get("locations", [])

                    for i, res in enumerate(results):
                        if i < len(payload_locations):
                            log = TrafficLog(
                                lat=payload_locations[i].get("lat"),
                                lng=payload_locations[i].get("lng"),
                                score=res.get("score"),
                                method=res.get("method"),
                                screenshot_url=res.get("screenshot_url"),
                                details=res,
                                job_id=job_record.id,
                            )
                            db.add(log)

                    await db.commit()
                    job["_logged_to_db"] = True
            except Exception as e:
                logger.warning(f"DB log failed for job {job_uid}: {e}")

    await job_queue.remove(job_uid)
    await update_job(
        db,
        job_uid,
        user.id,
        status=response.status,
        error=response.error,
    )

    return response


@app.post("/job/{job_uid}/cancel", response_model=TrafficResponse)
async def cancel_job(
    job_uid: str, user=Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    job = await job_queue.cancel(job_uid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = TrafficResponse(
        job_id=job_uid,
        status=job.get("status").value,
        completed=job.get("completed", 0),
        locations_count=job.get("payload", {}).get("count", 0),
        result=job.get("result"),
        error=job.get("error"),
    )

    await update_job(
        db,
        job_uid,
        user.id,
        status=response.status,
        error=response.error,
    )

    return response


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify Selenium Grid status
    """
    selenium_status = "unknown"
    grid_ready = False
    available_nodes = 0

    # Check Selenium Grid status
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://selenium-hub:4444/status")
            if response.status_code == 200:
                data = response.json()
                grid_ready = data.get("value", {}).get("ready", False)
                available_nodes = len(data.get("value", {}).get("nodes", []))
                selenium_status = "healthy" if grid_ready else "degraded"
            else:
                selenium_status = f"unhealthy: HTTP {response.status_code}"
    except Exception as e:
        selenium_status = f"unhealthy: {str(e)}"

    return {
        "api": "healthy",
        "selenium_grid": {
            "status": selenium_status,
            "ready": grid_ready,
            "nodes": available_nodes,
            "max_sessions": available_nodes * 4,
        },
    }
