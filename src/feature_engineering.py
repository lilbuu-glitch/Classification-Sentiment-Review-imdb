import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from .preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for text preprocessing in scikit-learn pipeline."""
    
    def __init__(self, remove_stopwords: bool = True, use_lemmatization: bool = True):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.preprocessor = TextPreprocessor(remove_stopwords, use_lemmatization)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        return self.preprocessor.preprocess_batch(X)

class FeatureEngineer:
    """Handles feature engineering for sentiment analysis."""
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 max_df: float = 0.9,
                 remove_stopwords: bool = True,
                 use_lemmatization: bool = True):
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        
    def create_pipeline(self) -> Pipeline:
        """
        Create a scikit-learn pipeline for feature engineering.
        
        Returns:
            Pipeline with preprocessing and vectorization steps
        """
        pipeline = Pipeline([
            ('preprocessor', TextPreprocessorTransformer(
                remove_stopwords=self.remove_stopwords,
                use_lemmatization=self.use_lemmatization
            )),
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words=None,  # We handle stopwords in preprocessing
                lowercase=False   # We handle lowercase in preprocessing
            ))
        ])
        
        self.pipeline = pipeline
        logger.info("Feature engineering pipeline created")
        return pipeline
    
    def fit_transform_features(self, X_train: pd.Series, y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the pipeline and transform training data.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            
        Returns:
            Tuple of (X_train_transformed, y_train_encoded)
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        # Transform text features
        X_train_transformed = self.pipeline.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        logger.info(f"Training features transformed. Shape: {X_train_transformed.shape}")
        logger.info(f"Classes encoded: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return X_train_transformed, y_train_encoded
    
    def transform_features(self, X: pd.Series, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using the fitted pipeline.
        
        Args:
            X: Text data
            y: Optional labels
            
        Returns:
            Tuple of (X_transformed, y_encoded)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform_features first.")
        
        # Transform text features
        X_transformed = self.pipeline.transform(X)
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
        
        logger.info(f"Features transformed. Shape: {X_transformed.shape}")
        return X_transformed, y_encoded
    
    def get_feature_names(self) -> np.ndarray:
        """
        Get feature names from the vectorizer.
        
        Returns:
            Array of feature names
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform_features first.")
        
        vectorizer = self.pipeline.named_steps['vectorizer']
        return vectorizer.get_feature_names_out()
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Vocabulary size
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform_features first.")
        
        vectorizer = self.pipeline.named_steps['vectorizer']
        return len(vectorizer.vocabulary_)
    
    def save_pipeline(self, filepath: str):
        """
        Save the feature engineering pipeline.
        
        Args:
            filepath: Path to save the pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform_features first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'params': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'remove_stopwords': self.remove_stopwords,
                'use_lemmatization': self.use_lemmatization
            }
        }, filepath)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """
        Load a saved feature engineering pipeline.
        
        Args:
            filepath: Path to the saved pipeline
        """
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.label_encoder = data['label_encoder']
        
        # Load parameters
        params = data['params']
        self.max_features = params['max_features']
        self.ngram_range = params['ngram_range']
        self.min_df = params['min_df']
        self.max_df = params['max_df']
        self.remove_stopwords = params['remove_stopwords']
        self.use_lemmatization = params['use_lemmatization']
        
        logger.info(f"Pipeline loaded from {filepath}")
    
    def get_feature_importance_info(self) -> Dict[str, Any]:
        """
        Get information about the feature engineering process.
        
        Returns:
            Dictionary with feature engineering statistics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform_features first.")
        
        vectorizer = self.pipeline.named_steps['vectorizer']
        
        info = {
            'vocabulary_size': len(vectorizer.vocabulary_),
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'remove_stopwords': self.remove_stopwords,
            'use_lemmatization': self.use_lemmatization,
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        
        return info

def main():
    """Main function to test the feature engineer."""
    # Example usage
    # Create sample data
    sample_texts = pd.Series([
        "This movie was great!",
        "I hated this film.",
        "Amazing acting and plot.",
        "Boring and predictable."
    ])
    sample_labels = pd.Series(["positive", "negative", "positive", "negative"])
    
    # Initialize feature engineer
    fe = FeatureEngineer(max_features=1000, ngram_range=(1, 2))
    
    # Fit and transform
    X_transformed, y_encoded = fe.fit_transform_features(sample_texts, sample_labels)
    
    print(f"Transformed features shape: {X_transformed.shape}")
    print(f"Encoded labels: {y_encoded}")
    print(f"Feature names (first 10): {fe.get_feature_names()[:10]}")
    print(f"Feature info: {fe.get_feature_importance_info()}")

if __name__ == "__main__":
    main()
