"""
FastAPI application for BBB penetration prediction.

Start the server with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or in production:
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:8000
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ErrorResponse,
)
from api.inference_service import InferenceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference service
inference_service = InferenceService()

# Model checkpoint path
MODEL_CHECKPOINT = Path(__file__).parent.parent / "models" / "checkpoints" / "best_model.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    Load model on startup, cleanup on shutdown.
    """
    # Startup
    logger.info("🚀 Starting BBB Prediction API...")
    if not inference_service.load_model(str(MODEL_CHECKPOINT)):
        logger.error("⚠️  Model failed to load. API will return errors.")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="BBB Penetration Predictor",
    description="Predict Blood-Brain Barrier penetration for molecules using transformer-based deep learning",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    """Custom OpenAPI schema with better documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="BBB Penetration Predictor API",
        version="1.0.0",
        description="REST API for predicting Blood-Brain Barrier penetration in molecules",
        routes=app.routes,
    )
    
    # Add example responses
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================================
# ENDPOINTS
# ============================================================================
# Serve static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/", tags=["System"])
async def serve_frontend():
    """Serve the HTML frontend."""
    return FileResponse("api/static/index.html")

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check"
)
async def health_check():
    """
    Check if the API and model are healthy and ready for predictions.
    
    **Returns:**
    - `status`: "healthy" or "degraded"
    - `model_loaded`: Whether the model is loaded
    - `model_name`: Name of the loaded model
    - `timestamp`: Current server time
    """
    return HealthResponse(
        status="healthy" if inference_service.is_ready() else "degraded",
        model_loaded=inference_service.is_ready(),
        model_name="SMILESXTransformer (BBB Penetration)",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predictions"],
    summary="Predict for single molecule",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid SMILES"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    }
)
async def predict_single(request: PredictRequest):
    """
    Predict BBB penetration for a single molecule.
    
    **Request:**
    - `smiles`: SMILES string (e.g., "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    
    **Returns:**
    - `bbb_score`: Probability of BBB penetration (0-1)
    - `prediction`: "BBB+" (penetrates) or "BBB-" (doesn't penetrate)
    - `confidence`: Model confidence in prediction
    - `rules_check`: Lipinski's rule violations if any
    
    **Example:**
```bash
    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: application/json" \\
      -d '{"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}'
```
    """
    if not inference_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check logs and restart the server."
        )
    
    try:
        result = inference_service.predict(request.smiles)
        
        if result.get("error"):
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )
        
        return PredictResponse(
            smiles=result["smiles"],
            bbb_score=result["bbb_score"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            rules_check=result["rules_check"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Predictions"],
    summary="Predict for multiple molecules",
    responses={
        200: {"description": "Batch predictions completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    }
)
async def predict_batch(request: BatchPredictRequest):
    """
    Predict BBB penetration for multiple molecules (batch inference).
    
    **Request:**
    - `molecules`: List of prediction requests (max 1000 molecules)
    
    **Returns:**
    - `predictions`: List of individual predictions
    - `batch_size`: Number of molecules processed
    - `processing_time_ms`: Total processing time
    
    **Example:**
```bash
    curl -X POST "http://localhost:8000/predict/batch" \\
      -H "Content-Type: application/json" \\
      -d '{
        "molecules": [
          {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
          {"smiles": "CCO"},
          {"smiles": "c1ccccc1"}
        ]
      }'
```
    """
    if not inference_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check logs and restart the server."
        )
    
    try:
        smiles_list = [mol.smiles for mol in request.molecules]
        results, processing_time = inference_service.batch_predict(smiles_list)
        
        predictions = []
        for result in results:
            if not result.get("error"):
                predictions.append(
                    PredictResponse(
                        smiles=result["smiles"],
                        bbb_score=result["bbb_score"],
                        prediction=result["prediction"],
                        confidence=result["confidence"],
                        rules_check=result["rules_check"],
                    )
                )
        
        return BatchPredictResponse(
            predictions=predictions,
            batch_size=len(smiles_list),
            processing_time_ms=round(processing_time, 2),
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/", tags=["System"])
async def root():
    """Welcome message with API info."""
    return {
        "message": "🧬 BBB Penetration Predictor API",
        "docs": "/docs (Swagger UI)",
        "redoc": "/redoc (ReDoc)",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch": "POST /predict/batch",
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=exc.status_code,
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )