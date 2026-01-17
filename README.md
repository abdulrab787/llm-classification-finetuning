# LLM Classification Fine-tuning

A comprehensive project for fine-tuning Large Language Models (LLMs) for text classification tasks.

## Project Structure

```
llm-classification-finetuning/
├── data/
│   ├── train.csv                 # Training data
│   ├── test.csv                  # Test data
│   ├── train_processed.csv       # Processed training data
│   ├── val_processed.csv         # Validation data
│   └── test_processed.csv        # Processed test data
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Data preprocessing
│   └── 03_model_baseline.ipynb   # Model training and evaluation
├── src/
│   ├── dataset.py               # Custom dataset class
│   ├── train.py                 # Training script
│   └── predict.py               # Prediction script
├── submissions/
│   └── submission.csv           # Model predictions
├── models/                      # Saved models directory
├── README.md                    # Project documentation
├── requirements.txt             # Project dependencies
└── .gitignore                   # Git ignore file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support, optional)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-classification-finetuning
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Exploratory Data Analysis
Start with the EDA notebook to understand your data:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Data Preprocessing
Process and prepare data for model training:
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### 3. Model Training
Train and evaluate the baseline model:
```bash
jupyter notebook notebooks/03_model_baseline.ipynb
```

Alternatively, use the training script:
```bash
python src/train.py
```

## Usage

### Training a Model

```python
from src.train import train_model

train_model(
    model_name="bert-base-uncased",
    train_path="data/train.csv",
    output_dir="models/my_model",
    num_epochs=3,
    batch_size=32,
    learning_rate=2e-5
)
```

### Making Predictions

```python
from src.predict import Predictor

predictor = Predictor(model_path="models/baseline")
predictions = predictor.predict(["Sample text 1", "Sample text 2"])
```

Or use the prediction script:
```bash
python src/predict.py
```

## Dataset Format

### Training Data (train.csv)
```
text,label
"Sample text here",positive
"Another sample",negative
...
```

### Test Data (test.csv)
```
text
"Test text 1"
"Test text 2"
...
```

## Configuration

### Hyperparameters
Key hyperparameters can be adjusted in the training scripts:
- `num_epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate for optimizer (default: 2e-5)
- `max_length`: Maximum sequence length (default: 128)

### Model Selection
You can use different pre-trained models:
- `bert-base-uncased`: BERT base model
- `distilbert-base-uncased`: DistilBERT (faster, smaller)
- `roberta-base`: RoBERTa base model
- Any other model from [Hugging Face Model Hub](https://huggingface.co/models)

## Results

The model evaluation metrics include:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Correct positive predictions / all positive predictions
- **Recall**: Correct positive predictions / all actual positives
- **F1-Score**: Harmonic mean of precision and recall

## Notebooks Overview

### 01_eda.ipynb
- Data loading and overview
- Missing value analysis
- Label distribution analysis
- Text statistics and visualization

### 02_preprocessing.ipynb
- Text cleaning (URLs, emails, special characters)
- Tokenization preparation
- Label encoding
- Train/validation split

### 03_model_baseline.ipynb
- Model initialization
- Custom dataset implementation
- Training loop with validation
- Performance evaluation
- Prediction generation
- Model saving

## Performance Optimization

### Memory Optimization
- Reduce `batch_size` if running out of memory
- Use `gradient_accumulation_steps` for effective larger batches
- Consider using `DistilBERT` for faster training

### Training Speed
- Use GPU acceleration (CUDA)
- Reduce `max_length` if texts are short
- Use mixed precision training (fp16)

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
batch_size = 16  # default is 32

# Or use gradient accumulation
gradient_accumulation_steps = 2
```

### Model not improving
- Increase training epochs
- Adjust learning rate (try 1e-5 or 5e-5)
- Ensure data is properly preprocessed
- Check for class imbalance in labels

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

## Contact

For questions or issues, please open an issue in the repository.
