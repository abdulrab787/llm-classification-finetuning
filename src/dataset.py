"""Dataset utilities for LLM classification."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import Dataset
import torch


class ClassificationDataset(Dataset):
    """Custom Dataset for text classification."""
    
    def __init__(
        self, 
        texts: list, 
        labels: Optional[list] = None,
        tokenizer=None,
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (optional)
            tokenizer: Tokenizer from transformers
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets."""
    from sklearn.model_selection import train_test_split
    
    train, val = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df.get('label', None) if 'label' in df.columns else None
    )
    
    return train, val
