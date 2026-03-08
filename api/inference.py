import logging
import time
import os
import sys
from typing import Tuple, Optional
import numpy as np
import joblib

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Handles sentiment prediction using the trained model."""
    
    def __init__(self, model_path: str = None, feature_pipeline_path: str = None):
        """
        Initialize the predictor with loaded model and feature pipeline.
        
        Args:
            model_path: Path to the saved model
            feature_pipeline_path: Path to the saved feature pipeline
        """
        self.model = None
        self.feature_engineer = None
        self.model_name = None
        self.label_encoder = None
        self.is_loaded = False
        
        if model_path and feature_pipeline_path:
            self.load_model(model_path, feature_pipeline_path)
    
    def load_model(self, model_path: str, feature_pipeline_path: str):
        """
        Load the trained model and feature pipeline.
        
        Args:
            model_path: Path to the saved model
            feature_pipeline_path: Path to the saved feature pipeline
        """
        try:
            # Load model data
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.feature_engineer = model_data['feature_engineer']
            self.label_encoder = self.feature_engineer.label_encoder
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")
            logger.info(f"Feature pipeline loaded from {feature_pipeline_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        start_time = time.time()
        
        try:
            # Transform the text using the feature pipeline
            text_features = self.feature_engineer.transform_features([text])[0]
            
            # Make prediction
            prediction = self.model.predict(text_features)[0]
            
            # Get confidence score
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_features)[0]
                confidence = max(probabilities)
            elif hasattr(self.model, 'decision_function'):
                decision_score = self.model.decision_function(text_features)[0]
                # Convert decision score to confidence-like score
                confidence = 1 / (1 + np.exp(-decision_score))
            else:
                confidence = 1.0  # Default confidence if no probability available
            
            # Convert numeric prediction to sentiment label
            sentiment = self.label_encoder.inverse_transform([prediction])[0]
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Prediction completed in {processing_time:.4f}s: {sentiment} ({confidence:.4f})")
            
            return sentiment, float(confidence)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(self, texts: list[str]) -> list[Tuple[str, float]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of tuples (sentiment, confidence)
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        start_time = time.time()
        results = []
        
        try:
            # Transform all texts at once for efficiency
            text_features = self.feature_engineer.transform_features(texts)[0]
            
            # Make predictions
            predictions = self.model.predict(text_features)
            
            # Get confidence scores
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_features)
                confidences = [max(prob) for prob in probabilities]
            elif hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(text_features)
                confidences = [1 / (1 + np.exp(-score)) for score in decision_scores]
            else:
                confidences = [1.0] * len(predictions)
            
            # Convert numeric predictions to sentiment labels
            sentiments = self.label_encoder.inverse_transform(predictions)
            
            # Combine results
            for sentiment, confidence in zip(sentiments, confidences):
                results.append((sentiment, float(confidence)))
            
            processing_time = time.time() - start_time
            logger.info(f"Batch prediction completed for {len(texts)} texts in {processing_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {
                "model_loaded": False,
                "error": "Model not loaded"
            }
        
        info = {
            "model_loaded": True,
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "feature_engineering": self.feature_engineer.get_feature_importance_info(),
            "classes": self.label_encoder.classes_.tolist(),
            "num_features": self.feature_engineer.get_vocabulary_size()
        }
        
        return info
    
    def health_check(self) -> dict:
        """
        Perform a health check on the predictor.
        
        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy" if self.is_loaded else "unhealthy",
            "model_loaded": self.is_loaded,
            "model_name": self.model_name if self.is_loaded else None,
            "timestamp": time.time()
        }
        
        if self.is_loaded:
            try:
                # Test prediction with a simple example
                test_text = "This is a test."
                sentiment, confidence = self.predict_single(test_text)
                health["test_prediction"] = {
                    "sentiment": sentiment,
                    "confidence": confidence
                }
            except Exception as e:
                health["status"] = "unhealthy"
                health["error"] = str(e)
        
        return health

# Global predictor instance
predictor = SentimentPredictor()

def get_predictor() -> SentimentPredictor:
    """Get the global predictor instance."""
    return predictor

def initialize_predictor(model_path: str = None, feature_pipeline_path: str = None):
    """
    Initialize the global predictor with model paths.
    
    Args:
        model_path: Path to the saved model
        feature_pipeline_path: Path to the saved feature pipeline
    """
    global predictor
    
    # Default paths if not provided
    if not model_path:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')
    if not feature_pipeline_path:
        feature_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_pipeline.joblib')
    
    predictor.load_model(model_path, feature_pipeline_path)
    logger.info("Global predictor initialized successfully")
