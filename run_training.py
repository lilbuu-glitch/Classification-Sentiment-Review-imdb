#!/usr/bin/env python3
"""
Run model training and evaluation without Jupyter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import time
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from train import ModelTrainer
from evaluate import ModelEvaluator
from utils import setup_logging, format_time

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

setup_logging("INFO")
logger = logging.getLogger(__name__)

def main():
    print("=== MODEL TRAINING AND EVALUATION ===")
    
    # 1. Load Prepared Data
    print("\n1. Loading prepared data...")
    try:
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/val.csv')
        test_df = pd.read_csv('data/test.csv')
        print("Data splits loaded successfully!")
    except FileNotFoundError:
        print("Data splits not found. Loading and preparing data...")
        loader = DataLoader('data/IMDB Dataset.csv')
        df = loader.load_data()
        df_clean = loader.clean_data()
        train_df, val_df, test_df = loader.split_data()
        loader.save_splits(train_df, val_df, test_df)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Display sentiment distribution
    print("\nSentiment distribution:")
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        sentiment_dist = split_df['sentiment'].value_counts()
        print(f"{split_name}: {sentiment_dist.to_dict()}")
    
    # 2. Feature Engineering
    print("\n2. Setting up feature engineering...")
    feature_engineer = FeatureEngineer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        remove_stopwords=True,
        use_lemmatization=True
    )
    
    print("Feature engineering parameters:")
    print(f"  Max features: {feature_engineer.max_features}")
    print(f"  N-gram range: {feature_engineer.ngram_range}")
    print(f"  Min document frequency: {feature_engineer.min_df}")
    print(f"  Max document frequency: {feature_engineer.max_df}")
    print(f"  Remove stopwords: {feature_engineer.remove_stopwords}")
    print(f"  Use lemmatization: {feature_engineer.use_lemmatization}")
    
    # Fit and transform training data
    print("\n3. Processing training data...")
    start_time = time.time()
    
    X_train, y_train = feature_engineer.fit_transform_features(
        train_df['review'], train_df['sentiment']
    )
    
    train_time = time.time() - start_time
    print(f"Training data processed in {format_time(train_time)}")
    print(f"Training features shape: {X_train.shape}")
    print(f"Vocabulary size: {feature_engineer.get_vocabulary_size()}")
    
    # Transform validation and test data
    print("\nProcessing validation data...")
    X_val, y_val = feature_engineer.transform_features(val_df['review'], val_df['sentiment'])
    print(f"Validation features shape: {X_val.shape}")
    
    print("\nProcessing test data...")
    X_test, y_test = feature_engineer.transform_features(test_df['review'], test_df['sentiment'])
    print(f"Test features shape: {X_test.shape}")
    
    # Display feature engineering information
    feature_info = feature_engineer.get_feature_importance_info()
    print("\nFeature Engineering Information:")
    for key, value in feature_info.items():
        print(f"  {key}: {value}")
    
    # Display sample feature names
    feature_names = feature_engineer.get_feature_names()
    print(f"\nSample feature names (first 10):")
    for i, feature in enumerate(feature_names[:10]):
        print(f"  {i+1:2d}. {feature}")
    
    # 4. Model Training
    print("\n4. Training models...")
    trainer = ModelTrainer(random_state=42)
    
    print("Available models:")
    for model_name in trainer.model_configs.keys():
        print(f"  - {model_name}")
    
    print("\nHyperparameter grids:")
    for model_name, config in trainer.model_configs.items():
        print(f"\n{model_name}:")
        for param, values in config['params'].items():
            print(f"  {param}: {values}")
    
    # Train all models
    print("\nStarting model training...")
    print("This may take some time depending on your system performance.")
    
    training_results = trainer.train_all_models(
        X_train, y_train, X_val, y_val, feature_engineer
    )
    
    print(f"\nTraining completed in {format_time(training_results['total_training_time'])}")
    print(f"Best model: {training_results['best_model_name']}")
    print(f"Best validation F1-score: {training_results['best_score']:.4f}")
    
    # Display model comparison
    comparison_df = training_results['comparison']
    print("\nModel Comparison:")
    print(comparison_df.round(4))
    
    # 5. Best Model Analysis
    print(f"\n5. Analyzing best model: {training_results['best_model_name'].upper()}")
    best_results = training_results['all_models'][training_results['best_model_name']]
    
    print(f"\nBest hyperparameters:")
    for param, value in best_results['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\nCross-validation scores:")
    print(f"  Mean F1-score: {best_results['cv_mean']:.4f}")
    print(f"  Std F1-score: {best_results['cv_std']:.4f}")
    print(f"  All CV scores: {[f'{score:.4f}' for score in best_results['cv_scores']]}")
    
    print(f"\nValidation metrics:")
    for metric, value in best_results['val_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTraining metrics:")
    for metric, value in best_results['train_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance analysis
    feature_importance_df = trainer.get_feature_importance(feature_names)
    
    if feature_importance_df is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10))
    else:
        print("\nFeature importance not available for this model type")
    
    # 6. Model Evaluation on Test Set
    print(f"\n6. Evaluating {training_results['best_model_name']} on test set...")
    evaluator = ModelEvaluator()
    
    test_results = evaluator.evaluate_model(
        trainer.best_model,
        X_test,
        y_test,
        model_name=training_results['best_model_name'],
        label_encoder=feature_engineer.label_encoder,
        save_plots=False,
        plot_dir='plots'
    )
    
    print(f"\nTest Results for {training_results['best_model_name']}:")
    for metric, value in test_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Display detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        test_results['true_labels'],
        test_results['predictions'],
        target_names=test_results['class_names']
    ))
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_results['true_labels'], test_results['predictions'])
    print(cm)
    
    if len(test_results['class_names']) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Details:")
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives: {tp}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nAdditional Metrics:")
        print(f"  Precision (Positive): {precision:.4f}")
        print(f"  Recall (Positive): {recall:.4f}")
        print(f"  Specificity (Negative): {specificity:.4f}")
    
    # 7. Save Model and Results
    print("\n7. Saving model and results...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Save best model
    model_path = 'models/best_model.joblib'
    trainer.save_best_model(model_path, feature_engineer)
    print(f"Best model saved to {model_path}")
    
    # Save feature pipeline
    pipeline_path = 'models/feature_pipeline.joblib'
    feature_engineer.save_pipeline(pipeline_path)
    print(f"Feature pipeline saved to {pipeline_path}")
    
    # Save evaluation results
    evaluator.save_all_results('evaluation_results')
    print("Evaluation results saved to evaluation_results/")
    
    # 8. Performance Summary
    performance_summary = {
        'best_model': {
            'name': training_results['best_model_name'],
            'hyperparameters': best_results['best_params'],
            'validation_f1': best_results['val_metrics']['f1_score'],
            'test_f1': test_results['metrics']['f1_score'],
            'test_accuracy': test_results['metrics']['accuracy'],
            'test_roc_auc': test_results['metrics'].get('roc_auc', 'N/A')
        },
        'feature_engineering': {
            'vocabulary_size': feature_engineer.get_vocabulary_size(),
            'max_features': feature_engineer.max_features,
            'ngram_range': feature_engineer.ngram_range,
            'preprocessing': {
                'remove_stopwords': feature_engineer.remove_stopwords,
                'use_lemmatization': feature_engineer.use_lemmatization
            }
        },
        'training_info': {
            'total_training_time': training_results['total_training_time'],
            'cross_validation_f1_mean': best_results['cv_mean'],
            'cross_validation_f1_std': best_results['cv_std']
        },
        'data_info': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'feature_dimensions': X_train.shape[1]
        }
    }
    
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(f"\nBest Model: {performance_summary['best_model']['name']}")
    print(f"Test Accuracy: {performance_summary['best_model']['test_accuracy']:.4f}")
    print(f"Test F1-Score: {performance_summary['best_model']['test_f1']:.4f}")
    if performance_summary['best_model']['test_roc_auc'] != 'N/A':
        print(f"Test ROC-AUC: {performance_summary['best_model']['test_roc_auc']:.4f}")
    
    print(f"\nFeature Engineering:")
    print(f"  Vocabulary Size: {performance_summary['feature_engineering']['vocabulary_size']}")
    print(f"  Feature Dimensions: {performance_summary['data_info']['feature_dimensions']}")
    print(f"  Preprocessing: Stopwords removed={performance_summary['feature_engineering']['preprocessing']['remove_stopwords']}, Lemmatization={performance_summary['feature_engineering']['preprocessing']['use_lemmatization']}")
    
    print(f"\nTraining:")
    print(f"  Total Time: {format_time(performance_summary['training_info']['total_training_time'])}")
    print(f"  CV F1-Score: {performance_summary['training_info']['cross_validation_f1_mean']:.4f} ± {performance_summary['training_info']['cross_validation_f1_std']:.4f}")
    
    print(f"\nData:")
    print(f"  Train/Val/Test samples: {performance_summary['data_info']['train_samples']}/{performance_summary['data_info']['val_samples']}/{performance_summary['data_info']['test_samples']}")
    
    print("\n=== CONCLUSION ===")
    if performance_summary['best_model']['test_f1'] > 0.85:
        print("✓ Excellent model performance achieved!")
    elif performance_summary['best_model']['test_f1'] > 0.80:
        print("✓ Good model performance achieved.")
    elif performance_summary['best_model']['test_f1'] > 0.75:
        print("✓ Acceptable model performance.")
    else:
        print("⚠ Model performance could be improved.")
    
    print("\nModel is ready for deployment!")

if __name__ == "__main__":
    import logging
    main()
