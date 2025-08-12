from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class StructuredText(BaseModel):
    product_name: Optional[str] = None
    ingredients: Optional[str] = None
    features: Optional[str] = None
    other_info: Optional[str] = None


class OCRPayload(BaseModel):
    full_text: str = ""
    structured_text: StructuredText | Dict[str, Any] = Field(default_factory=dict)


class DetectionObject(BaseModel):
    class_: str = Field(alias="class")
    bounding_box: List[int]
    confidence: float
    material_guess: str
    material_confidence: float
    low_confidence: bool

    class Config:
        allow_population_by_field_name = True


class QualityMetrics(BaseModel):
    blur: float
    brightness: float
    contrast: float


class LatencyBreakdown(BaseModel):
    validate: Optional[float] = 0.0
    preprocess: Optional[float] = 0.0
    inference: Optional[float] = 0.0
    postprocess: Optional[float] = 0.0


class OCRResponse(BaseModel):
    status: Literal["success", "fail"] | str
    engine_used: Optional[str] = None
    ocr: OCRPayload
    object_detection: List[DetectionObject] = Field(default_factory=list)
    quality: QualityMetrics
    latency: LatencyBreakdown
    restoration_attempted: bool = False
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    avg_blur: float = 0.0
    avg_brightness: float = 0.0
    avg_contrast: float = 0.0
    avg_confidence: float = 0.0
    avg_cpu: float = 0.0
    error_rate: float = 0.0
    avg_latency: float = 0.0
    latency_trend: List[float] = Field(default_factory=list)
    accuracy_trend: List[float] = Field(default_factory=list)
    cpu_trend: List[float] = Field(default_factory=list)
    count: int = 0


class HealthResponse(BaseModel):
    status: str
    cpu_percent: float
    rss_bytes: int


# --- Sustainability analysis models ---
class ExplanationItem(BaseModel):
    title: str
    explanation: str
    source_url: str


class Explanations(BaseModel):
    positives: List[ExplanationItem] = Field(default_factory=list)
    negatives: List[ExplanationItem] = Field(default_factory=list)


class SustainabilityResponse(BaseModel):
    product_name: str = ""
    date: str
    score: str = ""
    summary: str = ""
    positives: List[str] = Field(default_factory=list)
    negatives: List[str] = Field(default_factory=list)
    explanations: Explanations = Field(default_factory=Explanations)
    recommendations: List[str] = Field(default_factory=list)
    limited_analysis: bool = False
    error: Optional[str] = None
    # Do not include raw LLM traces or API keys in responses


class SustainabilityRequest(BaseModel):
    text: str
