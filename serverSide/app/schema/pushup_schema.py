from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PushupAnalysisRequest(BaseModel):
    """Request model for pushup analysis"""
    video_url: Optional[str] = None
    min_form_score: Optional[int] = 75
    model_name: Optional[str] = "GradientBoosting"


class PushupRepDetail(BaseModel):
    """Details of a single pushup rep"""
    rep_number: int
    form_score: float
    duration: Optional[float] = None
    timestamp: Optional[float] = None


class PushupAnalysisResponse(BaseModel):
    """Response model for pushup analysis"""
    status: str
    filename: Optional[str] = None
    pushups: int
    estimated_target: int
    average_form_score: Optional[float] = None
    rep_details: Optional[List[PushupRepDetail]] = None
    message: Optional[str] = None
    processing_time: Optional[float] = None
