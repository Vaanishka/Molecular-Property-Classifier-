"""
Pydantic schemas for API requests and responses.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single molecule prediction request."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            }
        }


class PredictResponse(BaseModel):
    """Single prediction response."""
    smiles: str
    bbb_score: float = Field(..., description="BBB penetration probability (0-1)")
    prediction: str = Field(..., description="'BBB+' or 'BBB-'")
    confidence: float = Field(..., description="Model confidence (0-1)")
    rules_check: Optional[Dict] = Field(None, description="Lipinski's rule violations if any")


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    molecules: List[PredictRequest] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "molecules": [
                    {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
                    {"smiles": "CCO"}
                ]
            }
        }


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictResponse]
    batch_size: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    details: Optional[str] = None
    code: int