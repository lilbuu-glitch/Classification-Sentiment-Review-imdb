import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
import joblib
import os

from .feature_engineering import FeatureEngineer
from .utils import setup_logging, ensure_dir, save_metrics, format_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and hyperparameter tuning."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.training_history = {}
        
        # Define models and their hyperparameter grids
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear']
                }
            },
            'linear_svm': {
                'model': LinearSVC(random_state=random_state, max_iter=2000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'loss': ['hinge']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            }
        }
        
    def train_single_model(self, 
                          model_name: str,
                          X_train, 
                          y_train,
                          X_val, 
                          y_val,
                          feature_engineer: FeatureEngineer) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_engineer: FeatureEngineer instance
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found in configurations")
        
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        config = self.model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Get probabilities for models that support it
        y_train_proba = None
        y_val_proba = None
        if hasattr(best_model, 'predict_proba'):
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_val_proba = best_model.predict_proba(X_val)[:, 1]
        elif hasattr(best_model, 'decision_function'):
            # For SVM, use decision function as proxy for probability
            y_train_proba = best_model.decision_function(X_train)
            y_val_proba = best_model.decision_function(X_val)
            # Normalize to [0, 1] range
            y_train_proba = (y_train_proba - y_train_proba.min()) / (y_train_proba.max() - y_train_proba.min())
            y_val_proba = (y_val_proba - y_val_proba.min()) / (y_val_proba.max() - y_val_proba.min())
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_weighted')
        
        training_time = time.time() - start_time
        
        results = {
            'model_name': model_name,
            'best_model': best_model,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'feature_engineer_info': feature_engineer.get_feature_importance_info()
        }
        
        self.models[model_name] = results
        self.training_history[model_name] = results
        
        # Update best model if this one is better
        if val_metrics['f1_score'] > self.best_score:
            self.best_model = best_model
            self.best_model_name = model_name
            self.best_score = val_metrics['f1_score']
            logger.info(f"New best model: {model_name} with F1-score: {self.best_score:.4f}")
        
        logger.info(f"{model_name} training completed in {format_time(training_time)}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"Validation F1-score: {val_metrics['f1_score']:.4f}")
        
        return results
    
    def train_all_models(self, 
                        X_train, 
                        y_train,
                        X_val, 
                        y_val,
                        feature_engineer: FeatureEngineer) -> Dict[str, Any]:
        """
        Train all models and compare results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_engineer: FeatureEngineer instance
            
        Returns:
            Dictionary with all training results
        """
        logger.info("Starting training for all models...")
        total_start_time = time.time()
        
        all_results = {}
        
        for model_name in self.model_configs.keys():
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, feature_engineer
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        total_time = time.time() - total_start_time
        
        # Create comparison summary
        comparison = self._create_model_comparison(all_results)
        
        final_results = {
            'all_models': all_results,
            'comparison': comparison,
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'total_training_time': total_time
        }
        
        logger.info(f"All models trained in {format_time(total_time)}")
        logger.info(f"Best model: {self.best_model_name} with F1-score: {self.best_score:.4f}")
        
        return final_results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add ROC-AUC if probabilities are available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                # Handle case where ROC-AUC cannot be calculated
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def _create_model_comparison(self, all_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a comparison DataFrame of all models."""
        comparison_data = []
        
        for model_name, results in all_results.items():
            if 'error' in results:
                continue
                
            row = {
                'model': model_name,
                'cv_mean_f1': results['cv_mean'],
                'cv_std_f1': results['cv_std'],
                'val_f1': results['val_metrics']['f1_score'],
                'val_accuracy': results['val_metrics']['accuracy'],
                'val_precision': results['val_metrics']['precision'],
                'val_recall': results['val_metrics']['recall'],
                'training_time': results['training_time']
            }
            
            if 'roc_auc' in results['val_metrics']:
                row['val_roc_auc'] = results['val_metrics']['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('val_f1', ascending=False)
        
        return comparison_df
    
    def save_best_model(self, filepath: str, feature_engineer: FeatureEngineer):
        """
        Save the best model and feature engineering pipeline.
        
        Args:
            filepath: Path to save the model
            feature_engineer: FeatureEngineer instance
        """
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        ensure_dir(os.path.dirname(filepath))
        
        # Save model and feature engineer together
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_engineer': feature_engineer,
            'best_score': self.best_score,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Best model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.best_score = model_data['best_score']
        self.training_history = model_data.get('training_history', {})
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model: {self.best_model_name}, Best score: {self.best_score:.4f}")
        
        return model_data['feature_engineer']
    
    def get_feature_importance(self, feature_names: np.ndarray) -> Optional[pd.DataFrame]:
        """
        Get feature importance for models that support it.
        
        Args:
            feature_names: Array of feature names
            
        Returns:
            DataFrame with feature importance or None if not supported
        """
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            # For tree-based models
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # For linear models
            importance = np.abs(self.best_model.coef_[0])
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df

def main():
    """Main function to test the model trainer."""
    # This would typically be used with actual data
    logger.info("ModelTrainer class ready for use")

if __name__ == "__main__":
    main()
