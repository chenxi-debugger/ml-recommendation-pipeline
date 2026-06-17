import pandera as pa
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