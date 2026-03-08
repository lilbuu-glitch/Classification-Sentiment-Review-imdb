#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from data_loader_simple import DataLoader
from preprocessing_simple import TextPreprocessor
from features_simple import FeaturePipeline
from trainer_simple import ModelTrainer

def main():
    print("Loading data...")
    loader = DataLoader('data/IMDB Dataset.csv')
    df = loader.load()
    df = loader.clean()
    
    print(f"Data shape: {df.shape}")
    
    print("Splitting data...")
    train_df, val_df, test_df = loader.split()
    loader.save_splits(train_df, val_df, test_df)
    
    print("Preprocessing...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    X_train_text = train_df['review'].apply(preprocessor.process)
    X_val_text = val_df['review'].apply(preprocessor.process)
    X_test_text = test_df['review'].apply(preprocessor.process)
    
    print("Feature engineering...")
    features = FeaturePipeline(max_features=5000)
    X_train, y_train = features.fit_transform(X_train_text, train_df['sentiment'])
    X_val, y_val = features.transform(X_val_text, val_df['sentiment'])
    X_test, y_test = features.transform(X_test_text, test_df['sentiment'])
    
    print(f"Feature shape: {X_train.shape}")
    
    print("Training models...")
    trainer = ModelTrainer()
    results = trainer.train_all(X_train, y_train, X_val, y_val)
    
    print("\nResults:")
    for name, result in results.items():
        print(f"{name}: CV={result['cv_score']:.3f}, Val F1={result['val_f1']:.3f}")
    
    print("Evaluating best model...")
    test_metrics = trainer.evaluate_best(X_test, y_test)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    trainer.save_best('models/best_model.joblib')
    features.save('models/features.joblib')
    
    print("Done!")

if __name__ == "__main__":
    main()
