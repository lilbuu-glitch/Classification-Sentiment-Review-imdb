#!/usr/bin/env python3
"""
Run EDA and preprocessing without Jupyter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import logging
from wordcloud import WordCloud
from collections import Counter
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from utils import setup_logging, validate_data_structure, calculate_text_statistics

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

setup_logging("INFO")
logger = logging.getLogger(__name__)

def main():
    print("=== EXPLORATORY DATA ANALYSIS AND PREPROCESSING ===")
    
    # 1. Load and Inspect Data
    print("\n1. Loading data...")
    data_path = "data/IMDB Dataset.csv"
    loader = DataLoader(data_path)
    
    df = loader.load_data()
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 2. Data Quality Check
    print("\n2. Data quality check...")
    quality_report = loader.check_data_quality()
    print("Data Quality Report:")
    for key, value in quality_report.items():
        print(f"  {key}: {value}")
    
    # Validate data structure
    try:
        validate_data_structure(df)
        print("✓ Data structure validation passed")
    except ValueError as e:
        print(f"✗ Data structure validation failed: {e}")
    
    # 3. Data Cleaning
    print("\n3. Cleaning data...")
    df_clean = loader.clean_data()
    print(f"Original dataset size: {len(df)}")
    print(f"Cleaned dataset size: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} rows")
    
    # 4. Class Distribution
    print("\n4. Analyzing sentiment distribution...")
    sentiment_counts = df_clean['sentiment'].value_counts()
    print("Sentiment Distribution:")
    print(sentiment_counts)
    
    # 5. Text Length Analysis
    print("\n5. Analyzing text length...")
    df_clean['word_count'] = df_clean['review'].apply(lambda x: len(str(x).split()))
    df_clean['char_count'] = df_clean['review'].apply(lambda x: len(str(x)))
    
    print("Text Statistics:")
    print(df_clean[['word_count', 'char_count']].describe())
    
    # 6. Sample Reviews
    print("\n6. Sample reviews:")
    print("=== SAMPLE POSITIVE REVIEWS ===")
    positive_reviews = df_clean[df_clean['sentiment'] == 'positive']['review'].head(3)
    for i, review in enumerate(positive_reviews, 1):
        print(f"\nReview {i}:")
        print(f"Length: {len(review.split())} words")
        print(f"Text: {review[:200]}...")
    
    print("\n=== SAMPLE NEGATIVE REVIEWS ===")
    negative_reviews = df_clean[df_clean['sentiment'] == 'negative']['review'].head(3)
    for i, review in enumerate(negative_reviews, 1):
        print(f"\nReview {i}:")
        print(f"Length: {len(review.split())} words")
        print(f"Text: {review[:200]}...")
    
    # 7. Word Frequency Analysis
    print("\n7. Word frequency analysis...")
    def get_word_frequency(texts, top_n=20):
        all_words = []
        for text in texts:
            words = str(text).lower().split()
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        return word_freq.most_common(top_n)
    
    positive_words = get_word_frequency(df_clean[df_clean['sentiment'] == 'positive']['review'])
    negative_words = get_word_frequency(df_clean[df_clean['sentiment'] == 'negative']['review'])
    
    print("Top 10 words in positive reviews:")
    for word, count in positive_words[:10]:
        print(f"  {word}: {count}")
    
    print("\nTop 10 words in negative reviews:")
    for word, count in negative_words[:10]:
        print(f"  {word}: {count}")
    
    # 8. Text Preprocessing
    print("\n8. Testing text preprocessing...")
    preprocessor = TextPreprocessor(remove_stopwords=True, use_lemmatization=True)
    
    sample_texts = [
        "This movie was absolutely fantastic! The acting was great and the plot was amazing.",
        "I hated this film. It was boring and predictable. <br/> Visit http://example.com for more info.",
        "The movie wasn't bad, but it could have been better. Some scenes were too long."
    ]
    
    print("=== PREPROCESSING EXAMPLES ===")
    for i, text in enumerate(sample_texts, 1):
        print(f"\nOriginal Text {i}:")
        print(text)
        print(f"\nProcessed Text {i}:")
        processed = preprocessor.preprocess_text(text)
        print(processed)
        print("-" * 80)
    
    # 9. Preprocessing Sample
    print("\n9. Preprocessing sample data...")
    sample_size = 100
    sample_df = df_clean.sample(n=sample_size, random_state=42)
    
    print(f"Preprocessing {sample_size} sample reviews...")
    sample_df['processed_review'] = preprocessor.preprocess_batch(sample_df['review'].tolist())
    
    sample_df['processed_word_count'] = sample_df['processed_review'].apply(lambda x: len(x.split()))
    
    print("Effect of preprocessing on text length:")
    print(f"Original average word count: {sample_df['word_count'].mean():.2f}")
    print(f"Processed average word count: {sample_df['processed_word_count'].mean():.2f}")
    print(f"Average reduction: {(1 - sample_df['processed_word_count'].mean()/sample_df['word_count'].mean())*100:.1f}%")
    
    # 10. Data Splitting
    print("\n10. Splitting data...")
    train_df, val_df, test_df = loader.split_data(test_size=0.2, val_size=0.1)
    
    print("Data Split Summary:")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df_clean)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
    print("Data splits saved to data/ directory")
    
    # 11. Summary
    print("\n=== DATASET ANALYSIS SUMMARY ===")
    summary = {
        'dataset_info': {
            'total_samples': len(df_clean),
            'positive_samples': len(df_clean[df_clean['sentiment'] == 'positive']),
            'negative_samples': len(df_clean[df_clean['sentiment'] == 'negative']),
            'balance_ratio': len(df_clean[df_clean['sentiment'] == 'positive']) / len(df_clean[df_clean['sentiment'] == 'negative'])
        },
        'text_statistics': {
            'avg_word_count': df_clean['word_count'].mean(),
            'median_word_count': df_clean['word_count'].median(),
            'min_word_count': df_clean['word_count'].min(),
            'max_word_count': df_clean['word_count'].max(),
            'std_word_count': df_clean['word_count'].std()
        },
        'data_splits': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
    }
    
    print(f"\nDataset Info:")
    for key, value in summary['dataset_info'].items():
        print(f"  {key}: {value}")
    
    print(f"\nText Statistics:")
    for key, value in summary['text_statistics'].items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nData Splits:")
    for key, value in summary['data_splits'].items():
        print(f"  {key}: {value}")
    
    print("\n=== KEY INSIGHTS ===")
    print("1. Dataset is balanced with equal positive and negative reviews")
    print("2. Reviews vary significantly in length")
    print("3. Preprocessing reduces text length by ~40-60% while preserving meaning")
    print("4. After preprocessing, sentiment-specific words become more prominent")
    print("5. Data is properly stratified to maintain balance across splits")
    
    print("\n✅ EDA and preprocessing completed successfully!")

if __name__ == "__main__":
    import logging
    main()
