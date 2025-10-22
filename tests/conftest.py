#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys

from sqlalchemy.util import md5_hex

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from db import Base, SessionLocal, engine
from models_db import User


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Insert default admin if not exists
    db = SessionLocal()
    if not db.query(User).filter_by(username="admin").first():
        db.add(User(username="admin", hashed_password=md5_hex("password123")))
        db.commit()
    db.close()

    yield

    # Teardown (optional)
    Base.metadata.drop_all(bind=engine)
