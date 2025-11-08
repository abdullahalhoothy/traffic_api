#!/usr/bin/python3
# -*- coding: utf-8 -*-

import asyncio
import concurrent.futures
import os
from contextlib import asynccontextmanager

from config import JOBQUEUE_MAX_JOBS, JOBQUEUE_PER_JOB_CONCURRENCY, logger
from db import Base, engine, get_db
from fastapi import FastAPI
from jobs import JobQueue
from models_db import User
from sqlalchemy import select
from sqlalchemy.util import md5_hex
from step2_traffic_analysis import analyze_location_traffic

# Thread pool for CPU-intensive operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)


async def run_single_location_blocking(
    lat,
    lng,
    storefront_direction,
    day_of_week,
    target_time,
    proxy=None,
):
    try:
        selenium_url = os.getenv("SELENIUM_URL", "http://selenium-hub:4444/wd/hub")
        result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            analyze_location_traffic,
            lat,
            lng,
            True,
            storefront_direction,
            day_of_week,
            target_time,
            selenium_url,
            proxy,
        )

        logger.info(f"Completed analysis for ({lat}, {lng}): Score {result['score']}")
        return result

    except Exception as e:
        logger.error(f"Analysis failed for ({lat}, {lng}): {e}")
        raise


job_queue: JobQueue = JobQueue(
    worker_callable=run_single_location_blocking,
    max_workers=JOBQUEUE_MAX_JOBS,
    per_job_concurrency=JOBQUEUE_PER_JOB_CONCURRENCY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create admin user
    async for db in get_db():
        admin_pw = os.getenv("ADMIN_PASSWORD", "123456").strip()
        result = await db.execute(select(User).filter_by(username="admin"))
        existing_admin = result.scalar_one_or_none()

        if not existing_admin:
            db.add(User(username="admin", hashed_password=md5_hex(admin_pw)))
            await db.commit()
        break

    # Start job queue
    await job_queue.start()
    logger.info("Job queue started")

    yield

    logger.info("Shutting down...")
    await job_queue.stop()
    thread_pool.shutdown(wait=True)
    logger.info("Cleanup completed")
