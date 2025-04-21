from typing import Optional
from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_output_tokens: int = Field(default=2048, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)