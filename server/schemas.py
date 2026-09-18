from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class InspectionKind(str, Enum):
    VISION = "vision"
    AUDIO = "audio"


class VisionInspectionResponse(BaseModel):
    kind: InspectionKind = InspectionKind.VISION
    filename: str
    anomaly_score: float = Field(..., description="Mean reconstruction error over the worst 5% of pixels")
    threshold: float
    is_defective: bool
    verdict: str
    heatmap_png_base64: str = Field(..., description="Original image with anomalous regions highlighted, as base64 PNG")


class AudioInspectionResponse(BaseModel):
    kind: InspectionKind = InspectionKind.AUDIO
    filename: str
    anomaly_score: float = Field(..., description="Mean reconstruction error over the worst 5% of spectrogram cells")
    threshold: float
    is_anomalous: bool
    verdict: str
    spectrogram_png_base64: str = Field(..., description="Mel-spectrogram with anomalous regions highlighted, as base64 PNG")


class HistoryEntry(BaseModel):
    id: int
    kind: InspectionKind
    filename: str
    verdict: str
    anomaly_score: float
    threshold: float
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    vision_model_loaded: bool
    audio_model_loaded: bool
