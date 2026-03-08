import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schema import (
    PredictionRequest, 
    PredictionResponse, 
    ErrorResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)
from .inference import get_predictor, initialize_predictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Sentiment Analysis API for IMDB Movie Reviews"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Sentiment Analysis API...")
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
        feature_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_pipeline.joblib')
        
        initialize_predictor(model_path, feature_pipeline_path)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    logger.info("Shutting down Sentiment Analysis API...")

app = FastAPI(
    title="Sentiment Analysis API",
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc) if exc.detail != str(exc) else None
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred"
        ).dict()
    )

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Sentiment Analysis API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        predictor = get_predictor()
        health = predictor.health_check()
        
        return HealthResponse(
            status=health["status"],
            model_loaded=health["model_loaded"],
            version=APP_VERSION
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version=APP_VERSION
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = predictor.get_model_info()
        
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type=info["model_type"],
            feature_engineering=info["feature_engineering"],
            performance_metrics=None,
            version=APP_VERSION
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        start_time = time.time()
        sentiment, confidence = predictor.predict_single(request.text)
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            sentiment=sentiment,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sentiment_batch(request: BatchPredictionRequest):
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        start_time = time.time()
        results = predictor.predict_batch(request.texts)
        total_time = time.time() - start_time
        
        predictions = []
        for i, (sentiment, confidence) in enumerate(results):
            predictions.append(PredictionResponse(
                sentiment=sentiment,
                confidence=confidence,
                processing_time=total_time / len(results)
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(results),
            processing_time=total_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )

@app.get("/stats")
async def get_api_stats():
    try:
        predictor = get_predictor()
        
        stats = {
            "api_version": APP_VERSION,
            "model_loaded": predictor.is_loaded,
            "model_name": predictor.model_name if predictor.is_loaded else None,
            "uptime": time.time(),
            "endpoints": {
                "predict": "/predict",
                "predict_batch": "/predict/batch",
                "model_info": "/model/info",
                "health": "/health",
                "docs": "/docs"
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API statistics"
        )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
