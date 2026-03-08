from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import re

class PredictionRequest(BaseModel):
    """Request schema for sentiment prediction."""
    
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="Text to analyze for sentiment"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate the input text."""
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        return v

class PredictionResponse(BaseModel):
    """Response schema for sentiment prediction."""
    
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.94,
                "processing_time": 0.023
            }
        }

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Text cannot be empty"
            }
        }

class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request schema for batch sentiment prediction."""
    
    texts: list[str] = Field(
        ..., 
        min_items=1, 
        max_items=100,
        description="List of texts to analyze for sentiment"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate the input texts."""
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        validated_texts = []
        for text in v:
            if not text or not text.strip():
                raise ValueError('Text cannot be empty')
            # Remove excessive whitespace
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            validated_texts.append(cleaned_text)
        
        return validated_texts

class BatchPredictionResponse(BaseModel):
    """Response schema for batch sentiment prediction."""
    
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of texts processed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "sentiment": "positive",
                        "confidence": 0.94,
                        "processing_time": 0.023
                    },
                    {
                        "sentiment": "negative",
                        "confidence": 0.87,
                        "processing_time": 0.018
                    }
                ],
                "total_processed": 2,
                "processing_time": 0.041
            }
        }

class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    
    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type of the model")
    feature_engineering: Dict[str, Any] = Field(..., description="Feature engineering configuration")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    version: str = Field(..., description="Model version")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "logistic_regression",
                "model_type": "LogisticRegression",
                "feature_engineering": {
                    "max_features": 10000,
                    "ngram_range": [1, 2],
                    "min_df": 5,
                    "max_df": 0.9
                },
                "performance_metrics": {
                    "accuracy": 0.89,
                    "f1_score": 0.89,
                    "roc_auc": 0.95
                },
                "version": "1.0.0"
            }
        }
