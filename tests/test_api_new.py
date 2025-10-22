#!/usr/bin/python3
# -*- coding: utf-8 -*-

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, job_queue


@pytest.mark.asyncio
async def test_login_and_token(setup_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://0.0.0.0:8000"
    ) as client:
        resp = await client.post(
            "/token",
            data={"username": "admin", "password": "password123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        assert token
        return token


@pytest.mark.asyncio
async def test_batch_submit_and_poll(mocker, setup_db):
    # Patch worker so it returns predictable result
    def fake_worker(lat, lng, *args, **kwargs):
        return {"score": lat + lng, "method": "mock"}

    job_queue.worker_callable = fake_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://0.0.0.0:8000"
    ) as client:
        # login
        resp = await client.post(
            "/token",
            data={"username": "admin", "password": "password123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # submit batch
        locations = [{"lat": i, "lng": i} for i in range(5)]
        resp2 = await client.post(
            "/analyze-batch", json={"locations": locations}, headers=headers
        )
        assert resp2.status_code == 200
        job_id = resp2.json()["job_id"]

        # poll until done
        for _ in range(10):
            r = await client.get(f"/job/{job_id}", headers=headers)
            data = r.json()
            if data["status"] == "done":
                results = data["result"]["results"]
                # ensure order preserved
                for i, res in enumerate(results):
                    assert res["score"] == i + i
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("Job did not complete in time")
