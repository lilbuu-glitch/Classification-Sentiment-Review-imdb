import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

SEED = 42
np.random.seed(SEED)

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None
        
    def load(self):
        self.data = pd.read_csv(self.path)
        return self.data
    
    def clean(self):
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()
        return self.data
    
    def split(self, test_size=0.2, val_size=0.1):
        train_val, test = train_test_split(
            self.data, test_size=test_size, random_state=SEED, stratify=self.data['sentiment']
        )
        val_adj = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_adj, random_state=SEED, stratify=train_val['sentiment']
        )
        return train, val, test
    
    def save_splits(self, train, val, test, out_dir="data"):
        os.makedirs(out_dir, exist_ok=True)
        train.to_csv(f"{out_dir}/train.csv", index=False)
        val.to_csv(f"{out_dir}/val.csv", index=False)
        test.to_csv(f"{out_dir}/test.csv", index=False)
