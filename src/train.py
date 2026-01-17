"""Training script for LLM classification."""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from dataset import ClassificationDataset, load_data, split_data
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    model_name: str = "bert-base-uncased",
    train_path: str = "../data/train.csv",
    output_dir: str = "../models",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
):
    """
    Train the classification model.
    
    Args:
        model_name: Pretrained model name from HuggingFace
        train_path: Path to training data
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    
    # Load data
    logger.info(f"Loading data from {train_path}")
    df = load_data(train_path)
    
    # Split data
    train_df, val_df = split_data(df, test_size=0.2)
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(df['label'].unique()) if 'label' in df.columns else 2
    )
    
    # Create datasets
    train_dataset = ClassificationDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    
    val_dataset = ClassificationDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    train_model()
