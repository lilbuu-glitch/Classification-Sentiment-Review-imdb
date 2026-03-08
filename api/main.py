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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application metadata
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Sentiment Analysis API for IMDB Movie Reviews"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Sentiment Analysis API...")
    
    try:
        # Initialize the predictor
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
        feature_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_pipeline.joblib')
        
        initialize_predictor(model_path, feature_pipeline_path)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue without model - endpoints will return appropriate errors
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment Analysis API...")

# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc) if exc.detail != str(exc) else None
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred"
        ).dict()
    )

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Sentiment Analysis API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
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
    """Get information about the loaded model."""
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
            performance_metrics=None,  # Could be loaded from a metrics file
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
    """
    Predict sentiment for a single text.
    
    - **text**: Text to analyze for sentiment (1-10000 characters)
    
    Returns:
    - **sentiment**: Predicted sentiment ("positive" or "negative")
    - **confidence**: Confidence score (0.0 to 1.0)
    - **processing_time**: Time taken to process the request (seconds)
    """
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Make prediction
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
    """
    Predict sentiment for multiple texts.
    
    - **texts**: List of texts to analyze (1-100 texts, each 1-10000 characters)
    
    Returns:
    - **predictions**: List of prediction results
    - **total_processed**: Total number of texts processed
    - **processing_time**: Total time taken to process all texts (seconds)
    """
    try:
        predictor = get_predictor()
        
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Make batch predictions
        start_time = time.time()
        results = predictor.predict_batch(request.texts)
        total_time = time.time() - start_time
        
        # Convert results to response format
        predictions = []
        for i, (sentiment, confidence) in enumerate(results):
            predictions.append(PredictionResponse(
                sentiment=sentiment,
                confidence=confidence,
                processing_time=total_time / len(results)  # Average time per prediction
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
    """Get basic API statistics."""
    try:
        predictor = get_predictor()
        
        stats = {
            "api_version": APP_VERSION,
            "model_loaded": predictor.is_loaded,
            "model_name": predictor.model_name if predictor.is_loaded else None,
            "uptime": time.time(),  # Could be improved to track actual uptime
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

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # For development only
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
