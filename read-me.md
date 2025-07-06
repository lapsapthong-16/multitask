# RoBERTa Training Pipeline for Sentiment and Emotion Classification

This repository contains a comprehensive pipeline for training RoBERTa models on sentiment and emotion classification tasks using Reddit data.

## Features

- **Dual-task training**: Separate models for sentiment and emotion classification
- **Pre-training**: Uses SST-2 for sentiment and GoEmotions for emotion classification
- **Comprehensive evaluation**: Includes accuracy, precision, recall, F1-score, Cohen's Kappa
- **Hyperparameter tuning**: Automated optimization of learning rate, batch size, epochs
- **Rich visualizations**: Confusion matrices, ROC curves, precision-recall curves, learning curves
- **Modular design**: Clean, well-documented code with separate functions

## Requirements

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. If you need additional dependencies:
```bash
python update_requirements.py
pip install -r requirements.txt
```

### Key Dependencies

- torch>=1.9.0
- transformers>=4.20.0
- datasets>=2.0.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- pandas>=1.3.0
- numpy>=1.21.0

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python run_roberta_training.py
```

### Advanced Usage

```bash
# Skip hyperparameter tuning (faster)
python run_roberta_training.py --no-tuning

# Use different model
python run_roberta_training.py --model roberta-large

# Specify custom data path
python run_roberta_training.py --data /path/to/your/data.csv

# Custom sequence length
python run_roberta_training.py --max-length 256
```

### Direct Python Usage

```python
from roberta_training import RobertaTrainer

# Initialize trainer
trainer = RobertaTrainer(model_name="roberta-base", max_length=512)

# Run complete pipeline
results = trainer.run_complete_pipeline(
    reddit_data_path='annotated_reddit_posts.csv',
    perform_hyperparameter_tuning=True
)
```

## Data Format

The input CSV file should have the following columns:
- `text_content`: The text to classify
- `sentiment_bertweet`: Ground truth sentiment labels
- `emotion_bertweet`: Ground truth emotion labels

Example:
```csv
id,text_content,sentiment_bertweet,emotion_bertweet
1,"I love this product!",Positive,Joy
2,"This is terrible",Negative,Anger
3,"It's okay",Neutral,Neutral
```

## Pipeline Overview

### 1. Data Loading and Preprocessing
- Loads Reddit dataset and external datasets (SST-2, GoEmotions)
- Tokenizes text using RoBERTa tokenizer
- Encodes labels and prepares datasets

### 2. Model Training
- **Sentiment Model**: Pre-trained on SST-2, then evaluated on Reddit data
- **Emotion Model**: Pre-trained on GoEmotions, then evaluated on Reddit data
- Separate training sessions for each task

### 3. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro and weighted precision
- **Recall**: Macro and weighted recall
- **F1-Score**: Macro and weighted F1-score
- **Cohen's Kappa**: Inter-rater agreement measure
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Per-class metrics

### 4. Hyperparameter Tuning
- **Parameters tuned**:
  - Learning rate: [1e-5, 2e-5, 3e-5, 5e-5]
  - Batch size: [8, 16, 32]
  - Number of epochs: [2, 3, 4]
  - Weight decay: [0.01, 0.1]
- Uses random search for efficiency
- Automatically re-trains with best parameters

### 5. Visualizations
- **Confusion Matrices**: For both sentiment and emotion
- **ROC Curves**: Multi-class ROC analysis
- **Precision-Recall Curves**: Detailed performance analysis
- **Learning Curves**: Training and validation loss/accuracy over epochs

## Output Files

### Models
- `roberta_sentiment_model_final/`: Final sentiment classification model
- `roberta_emotion_model_final/`: Final emotion classification model

### Plots
- `plots/sentiment_confusion_matrix.png`: Sentiment confusion matrix
- `plots/emotion_confusion_matrix.png`: Emotion confusion matrix
- `plots/sentiment_roc_curves.png`: Sentiment ROC curves
- `plots/emotion_roc_curves.png`: Emotion ROC curves
- `plots/sentiment_pr_curves.png`: Sentiment precision-recall curves
- `plots/emotion_pr_curves.png`: Emotion precision-recall curves
- `plots/sentiment_learning_curves.png`: Sentiment training curves
- `plots/emotion_learning_curves.png`: Emotion training curves

### Results
- `results/roberta_evaluation_summary.csv`: Summary of all metrics
- `results/roberta_complete_pipeline_*.json`: Detailed results in JSON format

## Architecture

### RobertaTrainer Class
Main class that orchestrates the entire pipeline:

- `setup_tokenizer()`: Initialize RoBERTa tokenizer
- `load_reddit_data()`: Load and validate Reddit dataset
- `load_external_datasets()`: Load SST-2 and GoEmotions datasets
- `train_model()`: Train RoBERTa model for specific task
- `evaluate_model()`: Comprehensive model evaluation
- `hyperparameter_tuning()`: Automated hyperparameter optimization
- `create_*_plots()`: Generate various visualizations
- `run_complete_pipeline()`: Execute full training pipeline

### RobertaDataset Class
Custom PyTorch Dataset for handling tokenized data:

- Handles text tokenization and label encoding
- Supports variable sequence lengths
- Optimized for batch processing

## Performance Expectations

### Typical Results
- **Sentiment Classification**: 75-85% accuracy
- **Emotion Classification**: 60-75% accuracy (more challenging due to more classes)

### Training Time
- **Without hyperparameter tuning**: ~30-60 minutes
- **With hyperparameter tuning**: ~2-4 hours
- Time varies based on hardware (GPU recommended)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 8`
   - Reduce sequence length: `--max-length