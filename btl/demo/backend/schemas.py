from pydantic import BaseModel
from typing import List

class PredictionResult(BaseModel):
    label: str
    confidence: float
    box: List[float] # [xmin, ymin, xmax, ymax]