"""Prediction script for LLM classification."""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Predictor class for making predictions with trained models."""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to use for inference
        """
        self.device = device
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def predict(self, texts: list, batch_size: int = 32) -> list:
        """
        Make predictions on a list of texts.
        
        Args:
            texts: List of text samples
            batch_size: Batch size for inference
            
        Returns:
            List of predicted labels
        """
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                predictions.extend(preds.tolist())
        
        return predictions


def predict_on_test(
    model_path: str,
    test_path: str,
    output_path: str
):
    """
    Make predictions on test data.
    
    Args:
        model_path: Path to trained model
        test_path: Path to test CSV file
        output_path: Path to save predictions
    """
    # Load test data
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Initialize predictor
    predictor = Predictor(model_path)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = predictor.predict(test_df['text'].tolist())
    
    # Save predictions
    test_df['prediction'] = predictions
    test_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    predict_on_test(
        model_path="../models",
        test_path="../data/test.csv",
        output_path="../submissions/submission.csv"
    )
