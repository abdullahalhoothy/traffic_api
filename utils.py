#!/usr/bin/python3
# -*- coding: utf-8 -*-

from config import logger
from models import TrafficResponse
from models_db import Job, TrafficLog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession


async def update_job(db: AsyncSession, job_id: str, user_id: int, **kwargs) -> None:
    try:
        stmt = (
            update(Job)
            .where(Job.uuid == job_id, Job.user_id == user_id)
            .values(**kwargs)
        )
        await db.execute(stmt)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to update job {job_id}: {e}")


async def get_job_record(
    db: AsyncSession, job_id: str, user_id: int
) -> TrafficResponse:
    try:
        result = await db.execute(
            select(Job).filter(Job.user_id == user_id, Job.uuid == job_id)
        )
        job_record = result.scalar_one_or_none()

        if job_record:
            result = await db.execute(
                select(TrafficLog).filter(TrafficLog.job_id == job_record.id)
            )
            traffic_logs = result.scalars().all()
            traffic_logs_count = len(traffic_logs)

            return TrafficResponse(
                job_id=job_id,
                status=job_record.status,
                completed=traffic_logs_count,
                locations_count=traffic_logs_count,
                result={
                    "count": traffic_logs_count,
                    "locations": [
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
                error=job_record.error,
            )

    except Exception as e:
        logger.warning(f"Failed to get job record {job_id}: {e}")
    return None
