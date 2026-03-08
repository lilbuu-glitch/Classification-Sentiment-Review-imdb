import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class DataLoader:
    """Handles loading and initial processing of the IMDB dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the IMDB dataset from CSV file."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check data quality and return statistics."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        quality_report = {
            'total_rows': len(self.data),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict()
        }
        
        # Check label distribution
        if 'sentiment' in self.data.columns:
            quality_report['label_distribution'] = self.data['sentiment'].value_counts().to_dict()
        
        logger.info("Data quality check completed")
        return quality_report
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset by removing duplicates and handling missing values."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        initial_rows = len(self.data)
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        duplicates_removed = initial_rows - len(self.data)
        
        # Handle missing values
        self.data = self.data.dropna()
        missing_removed = initial_rows - duplicates_removed - len(self.data)
        
        logger.info(f"Removed {duplicates_removed} duplicates and {missing_removed} rows with missing values")
        logger.info(f"Final dataset shape: {self.data.shape}")
        
        return self.data
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets using stratified splitting.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=RANDOM_SEED,
            stratify=self.data['sentiment']
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=RANDOM_SEED,
            stratify=train_val_df['sentiment']
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/len(self.data)*100:.1f}%)")
        logger.info(f"Validation set: {len(val_df)} samples ({len(val_df)/len(self.data)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(self.data)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data"):
        """Save the data splits to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        
        logger.info(f"Data splits saved to {output_dir}")

def main():
    """Main function to test the data loader."""
    # Example usage
    data_path = "data/IMDB Dataset.csv"
    loader = DataLoader(data_path)
    
    # Load and clean data
    data = loader.load_data()
    quality_report = loader.check_data_quality()
    print("Data Quality Report:")
    for key, value in quality_report.items():
        print(f"{key}: {value}")
    
    cleaned_data = loader.clean_data()
    
    # Split data
    train_df, val_df, test_df = loader.split_data()
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)

if __name__ == "__main__":
    main()
