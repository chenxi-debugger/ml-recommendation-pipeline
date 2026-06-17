import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, Field

# ===== 1. Model input validation (the 10-feature DataFrame) =====
class InputSchema(pa.DataFrameModel):
    hour: Series[int] = pa.Field(ge=0, le=23, coerce=True)
    day_of_week: Series[int] = pa.Field(ge=0, le=6, coerce=True)
    is_weekend: Series[int] = pa.Field(isin=[0, 1], coerce=True)
    user_activity: Series[float] = pa.Field(ge=0, coerce=True)
    video_popularity: Series[float] = pa.Field(ge=0, coerce=True)
    user_emb_0: Series[float] = pa.Field(coerce=True)
    user_emb_1: Series[float] = pa.Field(coerce=True)
    user_emb_2: Series[float] = pa.Field(coerce=True)
    user_emb_3: Series[float] = pa.Field(coerce=True)
    user_emb_4: Series[float] = pa.Field(coerce=True)

    class Config:
        strict = True

# ===== 2. Frontend request validation (only 4 fields, project.md 6.1) =====
class PredictRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    video_id: str = Field(..., description="Video ID")
    watch_time_seconds: float = Field(..., ge=0, description="Watch time in seconds, must be >= 0")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")

# ===== 3. Response returned to frontend (project.md 8.1) =====
class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Liked or not: 0/1")
    probability: float = Field(..., description="Like probability, 0-1")
    confidence: str = Field(..., description="Confidence level: High/Medium/Low")