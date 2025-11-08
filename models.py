#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TrafficResponse(BaseModel):
    job_id: str
    status: str
    completed: int
    locations_count: int
    result: Dict[str, Any]  # {"count": int, "locations": List[Dict[str, Any]]}
    error: Optional[str] = None


class LocationItem(BaseModel):
    lat: float
    lng: float
    storefront_direction: Optional[str] = "north"
    day: Optional[str] = None
    time: Optional[str] = None


class MultiTrafficRequest(BaseModel):
    locations: List[LocationItem]
    proxy: Optional[str] = None
