import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading, cleaning, and splitting for sentiment analysis."""
    
    def __init__(self, path: str):
        """
        Initialize DataLoader.
        
        Args:
            path: Path to the CSV dataset
        """
        self.path = path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV path.
        
        Returns:
            Loaded pandas DataFrame
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
            
        logger.info(f"Loading data from {self.path}...")
        self.data = pd.read_csv(self.path)
        logger.info(f"Loaded {len(self.data)} samples.")
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Perform basic data cleaning: remove duplicates and drop NaNs.
        
        Returns:
            Cleaned pandas DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        initial_count = len(self.data)
        
        # Drop duplicates
        self.data = self.data.drop_duplicates()
        # Drop rows with missing values
        self.data = self.data.dropna()
        
        final_count = len(self.data)
        logger.info(f"Data cleaning complete. Removed {initial_count - final_count} samples.")
        return self.data
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> tuple:
        """
        Split the data into train, val, and test sets.
        
        Args:
            test_size: Proportion for the test set
            val_size: Proportion for the validation set (relative to original total)
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() and clean_data() first.")
            
        logger.info(f"Splitting data (test_size={test_size}, val_size={val_size})...")
        
        # Split into train+val and test
        train_val, test = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.data['sentiment'] if 'sentiment' in self.data else None
        )
        
        # Calculate adjusted validation size for the second split
        # If val_size is 0.1 of total, and train_val is 0.8 of total:
        # val_adj = 0.1 / 0.8 = 0.125
        val_adj = val_size / (1 - test_size)
        
        train, val = train_test_split(
            train_val, 
            test_size=val_adj, 
            random_state=random_state, 
            stratify=train_val['sentiment'] if 'sentiment' in train_val else None
        )
        
        logger.info(f"Split sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test
    
    def save_splits(self, train, val, test, out_dir: str = "data"):
        """
        Save the data splits to CSV files.
        
        Args:
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            out_dir: Directory to save the files
        """
        os.makedirs(out_dir, exist_ok=True)
        
        train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
        
        logger.info(f"Data splits saved to {out_dir}/")

if __name__ == "__main__":
    # Simple test code
    import tempfile
    
    # Create dummy data for testing
    dummy_data = pd.DataFrame({
        'review': ['text1', 'text2', 'text3', 'text4', 'text5', 'text6', 'text7', 'text8', 'text9', 'text10'],
        'sentiment': ['pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg']
    })
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        dummy_data.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
        
    try:
        loader = DataLoader(tmp_path)
        loader.load_data()
        loader.clean_data()
        train, val, test = loader.split_data(test_size=0.2, val_size=0.2)
        print(f"Test split volumes: {len(train)}, {len(val)}, {len(test)}")
        loader.save_splits(train, val, test, "test_data_output")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
