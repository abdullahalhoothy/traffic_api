#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import os

SECRET_KEY = os.getenv("JWT_SECRET", "jwt_secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
RATE = os.getenv("RATE_LIMIT", "10/minute")

# JobQueue configuration
JOBQUEUE_MAX_JOBS = int(os.getenv("JOBQUEUE_MAX_JOBS", 1))  # concurrent jobs
JOBQUEUE_PER_JOB_CONCURRENCY = int(
    os.getenv("JOBQUEUE_PER_JOB_CONCURRENCY", 20)
)  # locations per job

# DataBase configuration
DB_FILE = os.getenv("SQLITE_DB_FILE", "traffic.db")
DB_URL = f"sqlite:///{DB_FILE}"

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traffic_api")
