#!/usr/bin/python3
# -*- coding: utf-8 -*-

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from main import app


@pytest.mark.asyncio
async def test_login_and_get_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8000"
    ) as client:
        resp = await client.post(
            "/token",
            data={"username": "admin", "password": "password123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data


@pytest.mark.asyncio
async def test_analyze_traffic_single(mocker):
    # Mock run_analysis_blocking to avoid Selenium
    fake_result = {
        "score": 42.5,
        "method": "mock",
        "screenshot_url": "/static/fake.png",
    }
    mocker.patch("main.run_analysis_blocking", return_value=fake_result)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8000"
    ) as client:
        # login
        resp = await client.post(
            "/token",
            data={"username": "admin", "password": "password123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # call analyze-traffic
        resp2 = await client.post(
            "/analyze-traffic",
            json={"lat": 37.77, "lng": -122.41, "save_to_static": False},
            headers=headers,
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["score"] == 42.5
        assert data["method"] == "mock"


@pytest.mark.asyncio
async def test_analyze_traffic_concurrent(mocker):
    fake_result = {
        "score": 99.9,
        "method": "mock",
        "screenshot_url": "/static/concurrent.png",
    }
    mocker.patch("main.run_analysis_blocking", return_value=fake_result)

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://localhost:8000"
    ) as client:
        # login
        resp = await client.post(
            "/token",
            data={"username": "admin", "password": "password123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # fire multiple requests concurrently
        tasks = [
            client.post(
                "/analyze-traffic",
                json={"lat": i, "lng": i, "save_to_static": False},
                headers=headers,
            )
            for i in range(5)
        ]
        responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)
        results = [r.json() for r in responses]
        assert all(res["score"] == 99.9 for res in results)
