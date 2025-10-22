#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import update
from sqlalchemy.util import md5_hex

from config import logger
from db import Base, engine, get_db
from models_db import Job, TrafficLog, User


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    admin_pw = os.getenv("ADMIN_PASSWORD", "123456").strip()
    if not db.query(User).filter_by(username="admin").first():
        db.add(User(username="admin", hashed_password=md5_hex(admin_pw)))
        db.commit()
    yield


def update_job(db, job_id: str, user_id: int, **kwargs) -> None:
    # db = next(get_db())

    try:
        stmt = (
            update(Job)
            .where(Job.uuid == job_id, Job.user_id == user_id)
            .values(**kwargs)
        )
        db.execute(stmt)
        db.commit()
    except Exception as e:
        logger.warning(f"DB log failed to update job {job_id}: {e}")


def get_job_record(db, job_id: str, user_id: int) -> dict | None:
    # db = next(get_db())

    try:
        job_record = (
            db.query(Job).filter(Job.user_id == user_id, Job.uuid == job_id).first()
        )
        if job_record:
            traffic_logs = (
                db.query(TrafficLog).filter(TrafficLog.job_id == job_record.id).all()
            )

            return {
                "job_id": job_id,
                "status": job_record.status,
                "remaining": job_record.remaining,
                "result": {
                    "count": len(traffic_logs),
                    "results": [
                        {
                            "lat": log.lat,
                            "lng": log.lng,
                            "score": log.score,
                            "method": log.method,
                            "screenshot_url": log.screenshot_url,
                            "details": log.details,
                        }
                        for log in traffic_logs
                    ],
                },
                "error": job_record.error,
            }
    except Exception:
        pass
