#!/usr/bin/env python3
"""
Training script for the sentiment analysis model.

This script handles the complete training pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training with hyperparameter tuning
4. Model evaluation
5. Saving the best model

Usage:
    python scripts/train_model.py [--config CONFIG_FILE]
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from train import ModelTrainer
from evaluate import ModelEvaluator
from utils import setup_logging, save_metrics, format_time

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/IMDB Dataset.csv',
        help='Path to the IMDB dataset CSV file'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='../models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='../evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='../plots',
        help='Directory to save plots'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=10000,
        help='Maximum number of features for TF-IDF'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proportion of training data for validation set'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting sentiment analysis model training")
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Training configuration
    config = {
        'max_features': args.max_features,
        'ngram_range': (1, 2),
        'min_df': 5,
        'max_df': 0.9,
        'remove_stopwords': True,
        'use_lemmatization': True,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'random_state': args.random_state
    }
    
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    start_time = time.time()
    
    try:
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        loader = DataLoader(args.data_path)
        
        # Load and clean data
        df = loader.load_data()
        df_clean = loader.clean_data()
        
        # Split data
        train_df, val_df, test_df = loader.split_data(
            test_size=config['test_size'],
            val_size=config['val_size']
        )
        
        # Save splits
        loader.save_splits(train_df, val_df, test_df)
        
        logger.info(f"Data prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering...")
        feature_engineer = FeatureEngineer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            remove_stopwords=config['remove_stopwords'],
            use_lemmatization=config['use_lemmatization']
        )
        
        # Fit and transform training data
        X_train, y_train = feature_engineer.fit_transform_features(
            train_df['review'], train_df['sentiment']
        )
        
        # Transform validation and test data
        X_val, y_val = feature_engineer.transform_features(val_df['review'], val_df['sentiment'])
        X_test, y_test = feature_engineer.transform_features(test_df['review'], test_df['sentiment'])
        
        logger.info(f"Feature engineering completed: {X_train.shape[1]} features")
        
        # Step 3: Model training
        logger.info("Step 3: Training models...")
        trainer = ModelTrainer(random_state=config['random_state'])
        
        # Train all models
        training_results = trainer.train_all_models(
            X_train, y_train, X_val, y_val, feature_engineer
        )
        
        logger.info(f"Training completed in {format_time(training_results['total_training_time'])}")
        logger.info(f"Best model: {training_results['best_model_name']} (F1: {training_results['best_score']:.4f})")
        
        # Step 4: Model evaluation
        logger.info("Step 4: Evaluating best model on test set...")
        evaluator = ModelEvaluator()
        
        # Evaluate best model
        test_results = evaluator.evaluate_model(
            trainer.best_model,
            X_test,
            y_test,
            model_name=training_results['best_model_name'],
            label_encoder=feature_engineer.label_encoder,
            save_plots=True,
            plot_dir=args.plots_dir
        )
        
        logger.info(f"Test F1-score: {test_results['metrics']['f1_score']:.4f}")
        logger.info(f"Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
        
        # Step 5: Save model and results
        logger.info("Step 5: Saving model and results...")
        
        # Save best model
        model_path = os.path.join(args.model_dir, 'best_model.joblib')
        trainer.save_best_model(model_path, feature_engineer)
        
        # Save feature pipeline
        pipeline_path = os.path.join(args.model_dir, 'feature_pipeline.joblib')
        feature_engineer.save_pipeline(pipeline_path)
        
        # Save evaluation results
        evaluator.save_all_results(args.results_dir)
        
        # Save training summary
        summary = {
            'config': config,
            'data_info': {
                'total_samples': len(df_clean),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df)
            },
            'feature_engineering': feature_engineer.get_feature_importance_info(),
            'training_results': {
                'best_model': training_results['best_model_name'],
                'best_cv_score': training_results['best_score'],
                'total_training_time': training_results['total_training_time'],
                'model_comparison': training_results['comparison'].to_dict()
            },
            'test_results': test_results['metrics'],
            'total_time': time.time() - start_time
        }
        
        summary_path = os.path.join(args.results_dir, 'training_summary.json')
        save_metrics(summary, summary_path)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Best model: {training_results['best_model_name']}")
        logger.info(f"Test F1-score: {test_results['metrics']['f1_score']:.4f}")
        logger.info(f"Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
        if 'roc_auc' in test_results['metrics']:
            logger.info(f"Test ROC-AUC: {test_results['metrics']['roc_auc']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {args.results_dir}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
