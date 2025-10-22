#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TrafficRequest(BaseModel):
    lat: float
    lng: float
    storefront_direction: str = "north"
    day: Optional[str] = None
    time: Optional[str] = None
    # save_to_static: bool = True


class TrafficResponse(BaseModel):
    score: float
    method: str
    screenshot_url: Optional[str] = None
    details: Dict[str, Any] = {}


class LocationItem(BaseModel):
    lat: float
    lng: float
    storefront_direction: Optional[str] = "north"
    day: Optional[str] = None
    time: Optional[str] = None
    # save_to_static: Optional[bool] = False


class MultiTrafficRequest(BaseModel):
    locations: List[LocationItem]
    proxy: Optional[str] = None
