#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Memory management
def aggressive_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    import gc
    gc.collect()
    print("🧹 Memory cleaned!")

print("✅ BERTweet Setup complete!")


# In[ ]:


class TrainingConfig:    
    def __init__(
        self,
        model_name: str = "vinai/bertweet-base",  
        max_length: int = 128,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        output_dir: str = "./bertweet_model_output",
        save_total_limit: int = 1,
        # Multi-task specific
        alpha: float = 0.5,  # Only used for multi-task
        task_type: str = "multitask"  # "sentiment", "emotion", "multitask"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.alpha = alpha
        self.task_type = task_type

class BERTweetModelConfig:
    
    def __init__(self):
        self.sentiment_classes = ['Negative', 'Neutral', 'Positive']
        self.emotion_classes = ['Anger', 'Fear', 'Joy', 'No Emotion', 'Sadness', 'Surprise']
        self.sentiment_num_classes = len(self.sentiment_classes)
        self.emotion_num_classes = len(self.emotion_classes)

bertweet_model_config = BERTweetModelConfig()
print("BERTweet Configuration classes defined!")


# In[ ]:


class BERTweetSingleTaskDataset(Dataset):
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # BERTweet specific preprocessing (handles tweets better)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }

class BERTweetMultiTaskDataset(Dataset):
    
    def __init__(
        self,
        texts: List[str],
        sentiment_labels: List[int],
        emotion_labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.emotion_labels = emotion_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts) == len(sentiment_labels) == len(emotion_labels), \
            "All inputs must have same length"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment_label = self.sentiment_labels[idx]
        emotion_label = self.emotion_labels[idx]
        
        # BERTweet specific preprocessing
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_labels': torch.tensor(sentiment_label, dtype=torch.long),
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long),
            'text': text
        }

print("BERTweet Dataset classes defined!")


# In[ ]:


class BERTweetSingleTaskTransformer(nn.Module):
    
    def __init__(
        self,
        model_name: str = "vinai/bertweet-base",
        num_classes: int = 3,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1
    ):
        super(BERTweetSingleTaskTransformer, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load BERTweet configuration
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_dropout_prob
        
        # BERTweet encoder
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head optimized for BERTweet
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),  # BERTweet uses GELU activation
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask):
        # BERTweet encoder output
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token for classification
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return {'logits': logits}
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "model_type": "BERTweetSingleTaskTransformer"
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"BERTweet single-task model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            **kwargs
        )
        
        # Load state dict
        model_file = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model

class BERTweetMultiTaskTransformer(nn.Module):
    
    def __init__(
        self,
        model_name: str = "vinai/bertweet-base",
        sentiment_num_classes: int = 3,
        emotion_num_classes: int = 6,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1
    ):
        super(BERTweetMultiTaskTransformer, self).__init__()
        
        self.model_name = model_name
        self.sentiment_num_classes = sentiment_num_classes
        self.emotion_num_classes = emotion_num_classes
        
        # Load BERTweet configuration
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_dropout_prob
        
        # Shared BERTweet encoder
        self.shared_encoder = AutoModel.from_pretrained(model_name, config=config)
        
        hidden_size = self.shared_encoder.config.hidden_size
        
        # Task-specific attention layers for BERTweet
        self.sentiment_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=12,  # BERTweet-base has 12 attention heads
            dropout=attention_dropout_prob,
            batch_first=True
        )
        
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=12,  # BERTweet-base has 12 attention heads
            dropout=attention_dropout_prob,
            batch_first=True
        )
        
        # Layer normalization
        self.sentiment_norm = nn.LayerNorm(hidden_size)
        self.emotion_norm = nn.LayerNorm(hidden_size)
        
        # Classification heads optimized for BERTweet
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size // 2, sentiment_num_classes)
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size // 2, emotion_num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.sentiment_classifier, self.emotion_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask):
        # Shared BERTweet encoder
        encoder_outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = encoder_outputs.last_hidden_state
        
        # Task-specific attention
        sentiment_attended, _ = self.sentiment_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        sentiment_attended = self.sentiment_norm(sentiment_attended + sequence_output)
        
        emotion_attended, _ = self.emotion_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        emotion_attended = self.emotion_norm(emotion_attended + sequence_output)
        
        # Use [CLS] token for classification
        sentiment_pooled = sentiment_attended[:, 0, :]
        emotion_pooled = emotion_attended[:, 0, :]
        
        # Classification
        sentiment_logits = self.sentiment_classifier(sentiment_pooled)
        emotion_logits = self.emotion_classifier(emotion_pooled)
        
        return {
            'sentiment_logits': sentiment_logits,
            'emotion_logits': emotion_logits
        }
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "sentiment_num_classes": self.sentiment_num_classes,
            "emotion_num_classes": self.emotion_num_classes,
            "model_type": "BERTweetMultiTaskTransformer"
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"BERTweet multi-task model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            model_name=config["model_name"],
            sentiment_num_classes=config["sentiment_num_classes"],
            emotion_num_classes=config["emotion_num_classes"],
            **kwargs
        )
        
        # Load state dict
        model_file = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model

print("BERTweet Model architectures defined!")


# In[ ]:


import joblib

def load_and_process_datasets_bertweet():
    
    print("Loading datasets for BERTweet...")
    
    # Load SST-2 for sentiment
    try:
        sst2_dataset = load_dataset("sst2")
        print(f"✅ SST-2 loaded: {len(sst2_dataset['train'])} train, {len(sst2_dataset['validation'])} val")
    except Exception as e:
        print(f"❌ Failed to load SST-2: {e}")
        return None, None
    
    # Load GoEmotion for emotion
    try:
        emotion_dataset = load_dataset("go_emotions", "simplified")
        print(f"✅ GoEmotion loaded: {len(emotion_dataset['train'])} train, {len(emotion_dataset['validation'])} val")
    except Exception as e:
        print(f"❌ Failed to load GoEmotion: {e}")
        return None, None
    
    # Try to load existing encoders first
    sentiment_encoder, emotion_encoder = load_existing_encoders_bertweet()
    
    # Process sentiment data (SST-2) for BERTweet
    sentiment_data = process_sentiment_data_bertweet(sst2_dataset, sentiment_encoder)
    
    # Process emotion data (GoEmotion) for BERTweet
    emotion_data = process_emotion_data_bertweet(emotion_dataset, emotion_encoder)
    
    return sentiment_data, emotion_data

def load_existing_encoders_bertweet():        
    try:
        sentiment_encoder = joblib.load('enc/sentiment_label_encoder.pkl')
        emotion_encoder = joblib.load('enc/emotion_label_encoder.pkl')
        print("✅ Loaded existing encoders from enc/ directory for BERTweet")
        return sentiment_encoder, emotion_encoder
    except Exception as e:
        print(f"⚠️ Could not load existing encoders: {e}")
        print("Creating new encoders for BERTweet...")
        
        # Create new encoders
        sentiment_encoder = LabelEncoder()
        emotion_encoder = LabelEncoder()
        sentiment_encoder.classes_ = np.array(bertweet_model_config.sentiment_classes)
        emotion_encoder.classes_ = np.array(bertweet_model_config.emotion_classes)
        
        # Save new encoders
        os.makedirs('enc', exist_ok=True)
        joblib.dump(sentiment_encoder, 'enc/bertweet_sentiment_label_encoder.pkl')
        joblib.dump(emotion_encoder, 'enc/bertweet_emotion_label_encoder.pkl')
        print("✅ Created and saved new BERTweet encoders")
        
        return sentiment_encoder, emotion_encoder

def process_sentiment_data_bertweet(sst2_dataset, sentiment_encoder, max_samples=None):    
    print("Processing sentiment data for BERTweet...")
    
    # Use full dataset if max_samples is None
    if max_samples is None:
        max_samples = len(sst2_dataset['train'])
    
    # Extract texts and labels
    train_texts = sst2_dataset['train']['sentence'][:max_samples]
    train_labels = sst2_dataset['train']['label'][:max_samples]
    
    val_texts = sst2_dataset['validation']['sentence']
    val_labels = sst2_dataset['validation']['label']
    
    # Map SST-2 labels to 3 classes: 0->Negative, 1->Positive
    # Add some neutral examples by random assignment
    expanded_labels = []
    expanded_texts = []
    
    for text, label in zip(train_texts, train_labels):
        if label == 0:  # Negative
            expanded_labels.append(0)
            expanded_texts.append(text)
        elif label == 1:  # Positive
            # Sometimes assign as positive, sometimes as neutral
            if np.random.random() < 0.15:  # 15% chance to be neutral
                expanded_labels.append(1)  # Neutral
            else:
                expanded_labels.append(2)  # Positive
            expanded_texts.append(text)
    
    # Ensure we have all 3 classes
    if 1 not in expanded_labels:
        # Force some examples to be neutral
        neutral_indices = np.random.choice(len(expanded_labels), size=100, replace=False)
        for idx in neutral_indices:
            expanded_labels[idx] = 1
    
    # Create train/val/test splits
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        expanded_texts, expanded_labels, test_size=0.3, random_state=42, stratify=expanded_labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    sentiment_data = {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels},
        'encoder': sentiment_encoder
    }
    
    print(f"BERTweet Sentiment data processed:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    return sentiment_data

def process_emotion_data_bertweet(emotion_dataset, emotion_encoder, max_samples=None):
    
    print("Processing emotion data for BERTweet...")
    
    # Filter to first 6 emotions only
    def filter_emotions(example):
        if isinstance(example['labels'], list):
            return example['labels'] and example['labels'][0] in range(6)
        else:
            return example['labels'] in range(6)
    
    filtered_train = emotion_dataset['train'].filter(filter_emotions)
    filtered_val = emotion_dataset['validation'].filter(filter_emotions)
    
    # Use full dataset if max_samples is None
    if max_samples is None:
        max_samples = len(filtered_train)
    
    # Extract texts and labels
    train_texts = filtered_train['text'][:max_samples]
    train_labels_raw = filtered_train['labels'][:max_samples]
    
    val_texts = filtered_val['text']
    val_labels_raw = filtered_val['labels']
    
    # Handle multi-label to single-label conversion
    train_labels = []
    for label in train_labels_raw:
        if isinstance(label, list):
            train_labels.append(label[0] if label else 0)
        else:
            train_labels.append(label)
    
    val_labels = []
    for label in val_labels_raw:
        if isinstance(label, list):
            val_labels.append(label[0] if label else 0)
        else:
            val_labels.append(label)
    
    # Create train/val/test splits
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        train_texts, train_labels, test_size=0.3, random_state=42, stratify=train_labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    emotion_data = {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels},
        'encoder': emotion_encoder
    }
    
    print(f"✅ BERTweet Emotion data processed:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    return emotion_data

def create_multitask_data_bertweet(sentiment_data, emotion_data):
    
    print("Creating multi-task dataset for BERTweet...")
    
    # Take minimum length to balance datasets
    min_train_len = min(len(sentiment_data['train']['texts']), len(emotion_data['train']['texts']))
    min_val_len = min(len(sentiment_data['val']['texts']), len(emotion_data['val']['texts']))
    min_test_len = min(len(sentiment_data['test']['texts']), len(emotion_data['test']['texts']))
    
    multitask_data = {
        'train': {
            'texts': sentiment_data['train']['texts'][:min_train_len],
            'sentiment_labels': sentiment_data['train']['labels'][:min_train_len],
            'emotion_labels': emotion_data['train']['labels'][:min_train_len]
        },
        'val': {
            'texts': sentiment_data['val']['texts'][:min_val_len],
            'sentiment_labels': sentiment_data['val']['labels'][:min_val_len],
            'emotion_labels': emotion_data['val']['labels'][:min_val_len]
        },
        'test': {
            'texts': sentiment_data['test']['texts'][:min_test_len],
            'sentiment_labels': sentiment_data['test']['labels'][:min_test_len],
            'emotion_labels': emotion_data['test']['labels'][:min_test_len]
        },
        'sentiment_encoder': sentiment_data['encoder'],
        'emotion_encoder': emotion_data['encoder']
    }
    
    print(f"BERTweet Multi-task data created:")
    print(f"  Train: {len(multitask_data['train']['texts'])} samples")
    print(f"  Val: {len(multitask_data['val']['texts'])} samples")
    print(f"  Test: {len(multitask_data['test']['texts'])} samples")
    
    return multitask_data

def load_reddit_evaluation_data():
    """Load Reddit data for evaluation"""
    print("Loading Reddit evaluation data...")
    
    try:
        # Load the annotated Reddit posts
        df = pd.read_csv('annotated_reddit_posts.csv')
        print(f"✅ Reddit data loaded: {len(df)} samples")
        
        # Create label encoders that match the model classes
        sentiment_encoder = LabelEncoder()
        emotion_encoder = LabelEncoder()
        
        # Fit encoders on Reddit data
        sentiment_encoder.fit(df['sentiment'].tolist())
        emotion_encoder.fit(df['emotion'].tolist())
        
        # Transform labels
        sentiment_labels = sentiment_encoder.transform(df['sentiment'].tolist())
        emotion_labels = emotion_encoder.transform(df['emotion'].tolist())
        
        # Create Reddit data in the format expected by evaluation functions
        reddit_data = {
            # For single-task sentiment evaluation
            'sentiment': {
                'texts': df['text_content'].tolist(),
                'labels': sentiment_labels,
                'labels_text': df['sentiment'].tolist()
            },
            # For single-task emotion evaluation
            'emotion': {
                'texts': df['text_content'].tolist(),
                'labels': emotion_labels,
                'labels_text': df['emotion'].tolist()
            },
            # For multitask evaluation
            'multitask': {
                'texts': df['text_content'].tolist(),
                'sentiment_labels': sentiment_labels,
                'emotion_labels': emotion_labels,
                'sentiment_labels_text': df['sentiment'].tolist(),
                'emotion_labels_text': df['emotion'].tolist()
            },
            # Keep encoders for reference
            'sentiment_encoder': sentiment_encoder,
            'emotion_encoder': emotion_encoder
        }
        
        print(f"✅ Reddit data prepared: {len(reddit_data['sentiment']['texts'])} samples")
        print(f"   Sentiment classes: {list(sentiment_encoder.classes_)}")
        print(f"   Emotion classes: {list(emotion_encoder.classes_)}")
        
        return reddit_data
        
    except Exception as e:
        print(f"❌ Error loading Reddit data: {e}")
        print("Falling back to empty Reddit data")
        return None

print("BERTweet Data processing functions defined!")


# In[ ]:


class BERTweetSingleTaskTrainer:
    
    def __init__(self, config: TrainingConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.device = device
        
        # Initialize BERTweet tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize BERTweet model
        self.model = BERTweetSingleTaskTransformer(
            model_name=config.model_name,
            num_classes=num_classes,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_dropout_prob=config.attention_dropout_prob,
            classifier_dropout=config.classifier_dropout
        ).to(self.device)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize tracking
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_macro': []
        }
    
    def create_data_loaders(self, data_splits: Dict):
        
        # Create datasets
        train_dataset = BERTweetSingleTaskDataset(
            texts=data_splits['train']['texts'],
            labels=data_splits['train']['labels'],
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        val_dataset = BERTweetSingleTaskDataset(
            texts=data_splits['val']['texts'],
            labels=data_splits['val']['labels'],
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler for BERTweet
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-6  # BERTweet specific epsilon
        )
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in self.train_loader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.loss_fn(outputs['logits'], labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self):
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs['logits'], labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1_macro
    
    def train(self, data_splits: Dict):
        print(f"Starting BERTweet single-task training ({self.config.task_type})...")
        
        # Setup data loaders
        self.create_data_loaders(data_splits)
        
        best_f1 = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            
            # Evaluate
            val_loss, val_accuracy, val_f1_macro = self.evaluate()
            
            # Track metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_f1_macro'].append(val_f1_macro)
            
            # Print results
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1_macro:.4f}")
            
            # Save best model
            if val_f1_macro > best_f1:
                best_f1 = val_f1_macro
                self.save_model(is_best=True)
        
        print(f"\n✅ BERTweet training completed! Best F1: {best_f1:.4f}")
        return self.training_history
    
    def save_model(self, is_best=False):
        suffix = "_best" if is_best else ""
        model_dir = os.path.join(self.config.output_dir, f"model{suffix}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        if is_best:
            print(f"Best BERTweet model saved to {model_dir}")

class BERTweetMultiTaskTrainer:
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = device
        
        # Initialize BERTweet tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize BERTweet multi-task model
        self.model = BERTweetMultiTaskTransformer(
            model_name=config.model_name,
            sentiment_num_classes=bertweet_model_config.sentiment_num_classes,
            emotion_num_classes=bertweet_model_config.emotion_num_classes,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_dropout_prob=config.attention_dropout_prob,
            classifier_dropout=config.classifier_dropout
        ).to(self.device)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize tracking
        self.training_history = {
            'train_loss': [],
            'train_sentiment_accuracy': [],
            'train_emotion_accuracy': [],
            'val_loss': [],
            'val_sentiment_accuracy': [],
            'val_emotion_accuracy': [],
            'val_sentiment_f1_macro': [],
            'val_emotion_f1_macro': []
        }
    
    def create_data_loaders(self, data_splits: Dict):
        
        # Create datasets
        train_dataset = BERTweetMultiTaskDataset(
            texts=data_splits['train']['texts'],
            sentiment_labels=data_splits['train']['sentiment_labels'],
            emotion_labels=data_splits['train']['emotion_labels'],
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        val_dataset = BERTweetMultiTaskDataset(
            texts=data_splits['val']['texts'],
            sentiment_labels=data_splits['val']['sentiment_labels'],
            emotion_labels=data_splits['val']['emotion_labels'],
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler for BERTweet
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-6  # BERTweet specific epsilon
        )
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0.0
        sentiment_correct = 0
        emotion_correct = 0
        total_predictions = 0
        
        for batch in self.train_loader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            sentiment_labels = batch['sentiment_labels'].to(self.device)
            emotion_labels = batch['emotion_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate losses
            sentiment_loss = self.loss_fn(outputs['sentiment_logits'], sentiment_labels)
            emotion_loss = self.loss_fn(outputs['emotion_logits'], emotion_labels)
            
            # Combined loss with alpha weighting
            loss = self.config.alpha * sentiment_loss + (1 - self.config.alpha) * emotion_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
            emotion_preds = torch.argmax(outputs['emotion_logits'], dim=-1)
            
            sentiment_correct += (sentiment_preds == sentiment_labels).sum().item()
            emotion_correct += (emotion_preds == emotion_labels).sum().item()
            total_predictions += sentiment_labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        sentiment_accuracy = sentiment_correct / total_predictions
        emotion_accuracy = emotion_correct / total_predictions
        
        return avg_loss, sentiment_accuracy, emotion_accuracy
    
    def evaluate(self):
        self.model.eval()
        
        total_loss = 0.0
        sentiment_predictions = []
        emotion_predictions = []
        sentiment_labels = []
        emotion_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sentiment_true = batch['sentiment_labels'].to(self.device)
                emotion_true = batch['emotion_labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                sentiment_loss = self.loss_fn(outputs['sentiment_logits'], sentiment_true)
                emotion_loss = self.loss_fn(outputs['emotion_logits'], emotion_true)
                loss = self.config.alpha * sentiment_loss + (1 - self.config.alpha) * emotion_loss
                
                total_loss += loss.item()
                
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
                emotion_preds = torch.argmax(outputs['emotion_logits'], dim=-1)
                
                sentiment_predictions.extend(sentiment_preds.cpu().numpy())
                emotion_predictions.extend(emotion_preds.cpu().numpy())
                sentiment_labels.extend(sentiment_true.cpu().numpy())
                emotion_labels.extend(emotion_true.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        sentiment_accuracy = accuracy_score(sentiment_labels, sentiment_predictions)
        emotion_accuracy = accuracy_score(emotion_labels, emotion_predictions)
        sentiment_f1_macro = f1_score(sentiment_labels, sentiment_predictions, average='macro', zero_division=0)
        emotion_f1_macro = f1_score(emotion_labels, emotion_predictions, average='macro', zero_division=0)
        
        return avg_loss, sentiment_accuracy, emotion_accuracy, sentiment_f1_macro, emotion_f1_macro
    
    def train(self, data_splits: Dict):
        print(f"Starting BERTweet multi-task training...")
        
        # Setup data loaders
        self.create_data_loaders(data_splits)
        
        best_combined_f1 = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_sent_acc, train_emo_acc = self.train_epoch()
            
            # Evaluate
            val_loss, val_sent_acc, val_emo_acc, val_sent_f1, val_emo_f1 = self.evaluate()
            
            # Track metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_sentiment_accuracy'].append(train_sent_acc)
            self.training_history['train_emotion_accuracy'].append(train_emo_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_sentiment_accuracy'].append(val_sent_acc)
            self.training_history['val_emotion_accuracy'].append(val_emo_acc)
            self.training_history['val_sentiment_f1_macro'].append(val_sent_f1)
            self.training_history['val_emotion_f1_macro'].append(val_emo_f1)
            
            # Print results
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Sentiment Acc: {train_sent_acc:.4f}, Train Emotion Acc: {train_emo_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Sentiment Acc: {val_sent_acc:.4f}, F1: {val_sent_f1:.4f}")
            print(f"  Val Emotion Acc: {val_emo_acc:.4f}, F1: {val_emo_f1:.4f}")
            
            # Save best model
            combined_f1 = (val_sent_f1 + val_emo_f1) / 2
            if combined_f1 > best_combined_f1:
                best_combined_f1 = combined_f1
                self.save_model(is_best=True)
        
        print(f"\nBERTweet training completed! Best Combined F1: {best_combined_f1:.4f}")
        return self.training_history
    
    def save_model(self, is_best=False):
        suffix = "_best" if is_best else ""
        model_dir = os.path.join(self.config.output_dir, f"model{suffix}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        if is_best:
            print(f"Best BERTweet model saved to {model_dir}")

print("BERTweet Training classes defined!")


# In[ ]:


def train_with_pruning(self, data_splits: Dict, trial=None):
    print(f"Starting BERTweet single-task training ({self.config.task_type})...")
    
    # Setup data loaders
    self.create_data_loaders(data_splits)
    
    best_f1 = 0.0
    
    for epoch in range(self.config.num_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
        
        # Train
        train_loss, train_accuracy = self.train_epoch()
        
        # Evaluate
        val_loss, val_accuracy, val_f1_macro = self.evaluate()
        
        # Track metrics
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_accuracy'].append(train_accuracy)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_accuracy'].append(val_accuracy)
        self.training_history['val_f1_macro'].append(val_f1_macro)
        
        # Print results
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1_macro:.4f}")
        
        # Report to ASHA for pruning decision
        if trial is not None:
            trial.report(val_f1_macro, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                print(f"  🚫 Trial pruned at epoch {epoch + 1}")
                raise optuna.TrialPruned()
        
        # Save best model
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            self.save_model(is_best=True)
    
    print(f"\nBERTweet training completed! Best F1: {best_f1:.4f}")
    return self.training_history


# In[ ]:


import time
import numpy as np
import optuna
from typing import Dict

def create_tuning_subset(data_splits, subset_ratio=0.03):
    print(f"🔪 Creating {subset_ratio*100:.0f}% subset for hyperparameter tuning...")
    
    def sample_split(split_data, ratio):
        n_samples = int(len(split_data['texts']) * ratio)
        if n_samples < 50:  # Ensure minimum samples
            n_samples = min(50, len(split_data['texts']))
        indices = np.random.choice(len(split_data['texts']), n_samples, replace=False)
        
        return {
            'texts': [split_data['texts'][i] for i in indices],
            'labels': [split_data['labels'][i] for i in indices]
        }
    
    # Handle different possible key names for validation set
    val_key = 'val' if 'val' in data_splits else ('validation' if 'validation' in data_splits else 'test')
    
    tuning_data = {
        'train': sample_split(data_splits['train'], subset_ratio),
        'val': sample_split(data_splits[val_key], subset_ratio),
        'test': sample_split(data_splits['test'], subset_ratio) if 'test' in data_splits else sample_split(data_splits[val_key], subset_ratio)
    }
    
    print(f"📊 Tuning subset created:")
    print(f"  Train: {len(tuning_data['train']['texts'])} samples")
    print(f"  Val: {len(tuning_data['val']['texts'])} samples")
    
    return tuning_data

def create_multitask_tuning_subset(data_splits, subset_ratio=0.03):
    print(f"Creating {subset_ratio*100:.0f}% multitask subset for hyperparameter tuning...")
    
    def sample_multitask_split(split_data, ratio):
        n_samples = int(len(split_data['texts']) * ratio)
        if n_samples < 50:  # Ensure minimum samples
            n_samples = min(50, len(split_data['texts']))
        indices = np.random.choice(len(split_data['texts']), n_samples, replace=False)
        
        return {
            'texts': [split_data['texts'][i] for i in indices],
            'sentiment_labels': [split_data['sentiment_labels'][i] for i in indices],
            'emotion_labels': [split_data['emotion_labels'][i] for i in indices]
        }
    
    val_key = 'val' if 'val' in data_splits else ('validation' if 'validation' in data_splits else 'test')
    
    tuning_data = {
        'train': sample_multitask_split(data_splits['train'], subset_ratio),
        'val': sample_multitask_split(data_splits[val_key], subset_ratio),
        'test': sample_multitask_split(data_splits['test'], subset_ratio) if 'test' in data_splits else sample_multitask_split(data_splits[val_key], subset_ratio)
    }
    
    print(f"📊 Multitask tuning subset created:")
    print(f"  Train: {len(tuning_data['train']['texts'])} samples")
    print(f"  Val: {len(tuning_data['val']['texts'])} samples")
    
    return tuning_data

class FastBERTweetHyperparameterTuner:
    
    def __init__(
        self,
        model_type: str,
        data_splits: Dict,
        n_trials: int = 8,
        model_name: str = "vinai/bertweet-base",
        subset_ratio: float = 0.03,
        max_epochs_per_trial: int = 2
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.model_name = model_name
        self.max_epochs_per_trial = max_epochs_per_trial
        
        print(f"🚀 Creating ultra-fast tuning setup for {model_type}...")
        
        if model_type == "multitask":
            self.tuning_data = create_multitask_tuning_subset(data_splits, subset_ratio)
        else:
            self.tuning_data = create_tuning_subset(data_splits, subset_ratio)
        
        print(f"⚡ Speed optimizations:")
        print(f"  - Using {subset_ratio*100:.0f}% of data ({len(self.tuning_data['train']['texts'])} samples)")
        print(f"  - Max {max_epochs_per_trial} epochs per trial")
        print(f"  - {n_trials} total trials")
        print(f"  - Estimated time: {n_trials * max_epochs_per_trial * 1:.0f}-{n_trials * max_epochs_per_trial * 3:.0f} minutes")
    
    def objective(self, trial):
        
        # Fast hyperparameter suggestions
        learning_rate = trial.suggest_float('learning_rate', 2e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        num_epochs = self.max_epochs_per_trial
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.1, 0.2)
        weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
        hidden_dropout = trial.suggest_float('hidden_dropout_prob', 0.1, 0.3)
        classifier_dropout = trial.suggest_float('classifier_dropout', 0.1, 0.3)
        max_length = 128
        
        alpha = trial.suggest_float('alpha', 0.4, 0.6) if self.model_type == "multitask" else 0.5
        
        # Create speed-optimized config (removed unsupported parameters)
        config = TrainingConfig(
            model_name=self.model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            hidden_dropout_prob=hidden_dropout,
            classifier_dropout=classifier_dropout,
            max_length=max_length,
            alpha=alpha,
            task_type=self.model_type,
            output_dir=f"./fast_trial_{trial.number}"
        )
        
        start_time = time.time()
        
        try:
            # Clear memory
            aggressive_memory_cleanup()
            
            if self.model_type == "multitask":
                trainer = BERTweetMultiTaskTrainer(config)
                history = trainer.train(self.tuning_data)
                
                # Get scores from general dataset (not Reddit)
                best_sentiment_f1 = max(history['val_sentiment_f1_macro']) if history['val_sentiment_f1_macro'] else 0.0
                best_emotion_f1 = max(history['val_emotion_f1_macro']) if history['val_emotion_f1_macro'] else 0.0
                score = (best_sentiment_f1 + best_emotion_f1) / 2
                
            else:
                if self.model_type == "sentiment":
                    num_classes = bertweet_model_config.sentiment_num_classes
                else:
                    num_classes = bertweet_model_config.emotion_num_classes
                
                trainer = BERTweetSingleTaskTrainer(config, num_classes)
                history = trainer.train(self.tuning_data)
                
                # Get score from general dataset (not Reddit)
                score = max(history['val_f1_macro']) if history['val_f1_macro'] else 0.0
            
            elapsed = time.time() - start_time
            print(f"⚡ Trial {trial.number}: Score={score:.4f}, Time={elapsed/60:.1f}min")
            
            return score
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Trial {trial.number} failed after {elapsed/60:.1f}min: {str(e)[:100]}...")
            return 0.0
            
        finally:
            # Cleanup
            if 'trainer' in locals():
                del trainer
            aggressive_memory_cleanup()
    
    def tune(self):
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.RandomSampler(seed=42)
        )
        
        print(f"\n🚀 Starting FAST hyperparameter tuning for {self.model_type}...")
        print(f"⚡ Target: Find good hyperparameters in ~{self.n_trials * 2:.0f} minutes")
        print("=" * 60)
        
        start_time = time.time()
        study.optimize(self.objective, n_trials=self.n_trials)
        total_time = time.time() - start_time
        
        print(f"\n🏆 Fast tuning completed in {total_time/60:.1f} minutes!")
        print(f"🎯 Best score: {study.best_value:.4f}")
        print(f"📋 Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        return study


# In[ ]:


def evaluate_bertweet_model(model_path: str, model_type: str, test_data: Dict, model_name: str = "vinai/bertweet-base", reddit_data: Dict = None):
    
    print(f"📊 Evaluating BERTweet {model_type} model...")
    
    # Load BERTweet tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load BERTweet model
    if model_type == "multitask":
        model = BERTweetMultiTaskTransformer.from_pretrained(model_path)
    else:
        model = BERTweetSingleTaskTransformer.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    # Evaluate on general dataset
    general_results = evaluate_bertweet_on_dataset(model, model_type, test_data, tokenizer, "General Dataset")
    
    # Evaluate on Reddit dataset if available
    reddit_results = None
    if reddit_data is not None:
        reddit_results = evaluate_bertweet_on_dataset(model, model_type, reddit_data, tokenizer, "Reddit Dataset")
    
    return {
        'general': general_results,
        'reddit': reddit_results
    }

def evaluate_bertweet_on_dataset(model, model_type: str, data: Dict, tokenizer, dataset_name: str):
    """Evaluate BERTweet model on a specific dataset"""
    print(f"Evaluating on {dataset_name}...")
    
    # Prepare test data
    if model_type == "multitask":
        test_dataset = BERTweetMultiTaskDataset(
            texts=data['texts'],
            sentiment_labels=data['sentiment_labels'],
            emotion_labels=data['emotion_labels'],
            tokenizer=tokenizer,
            max_length=128
        )
    else:
        test_dataset = BERTweetSingleTaskDataset(
            texts=data['texts'],
            labels=data['labels'],
            tokenizer=tokenizer,
            max_length=128
        )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate
    if model_type == "multitask":
        all_sentiment_predictions = []
        all_emotion_predictions = []
        all_sentiment_labels = []
        all_emotion_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
                emotion_preds = torch.argmax(outputs['emotion_logits'], dim=-1)
                
                all_sentiment_predictions.extend(sentiment_preds.cpu().numpy())
                all_emotion_predictions.extend(emotion_preds.cpu().numpy())
                all_sentiment_labels.extend(batch['sentiment_labels'].numpy())
                all_emotion_labels.extend(batch['emotion_labels'].numpy())
        
        # Calculate metrics
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_predictions)
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_predictions)
        sentiment_f1_macro = f1_score(all_sentiment_labels, all_sentiment_predictions, average='macro', zero_division=0)
        emotion_f1_macro = f1_score(all_emotion_labels, all_emotion_predictions, average='macro', zero_division=0)
        
        results = {
            'sentiment_accuracy': sentiment_accuracy,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_f1_macro': sentiment_f1_macro,
            'emotion_f1_macro': emotion_f1_macro,
            'combined_accuracy': (sentiment_accuracy + emotion_accuracy) / 2,
            'combined_f1_macro': (sentiment_f1_macro + emotion_f1_macro) / 2
        }
        
        print(f"📊 BERTweet Multi-task Results on {dataset_name}:")
        print(f"  Sentiment - Accuracy: {sentiment_accuracy:.4f}, F1: {sentiment_f1_macro:.4f}")
        print(f"  Emotion - Accuracy: {emotion_accuracy:.4f}, F1: {emotion_f1_macro:.4f}")
        print(f"  Combined - Accuracy: {results['combined_accuracy']:.4f}, F1: {results['combined_f1_macro']:.4f}")
        
    else:
        # Single-task evaluation
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
        
        print(f"📊 BERTweet {model_type.capitalize()} Results on {dataset_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
    
    return results

def create_bertweet_results_summary(sentiment_results: Dict, emotion_results: Dict, multitask_results: Dict):
    
    print(f"\n" + "="*80)
    print(f"📊 BERTWEET FINAL RESULTS SUMMARY")
    print(f"="*80)
    
    print(f"\n🎯 BERTWEET SINGLE-TASK SENTIMENT MODEL:")
    print(f"  Accuracy: {sentiment_results['accuracy']:.4f}")
    print(f"  F1 Macro: {sentiment_results['f1_macro']:.4f}")
    
    print(f"\n😊 BERTWEET SINGLE-TASK EMOTION MODEL:")
    print(f"  Accuracy: {emotion_results['accuracy']:.4f}")
    print(f"  F1 Macro: {emotion_results['f1_macro']:.4f}")
    
    print(f"\n🔗 BERTWEET MULTI-TASK MODEL:")
    print(f"  Sentiment - Accuracy: {multitask_results['sentiment_accuracy']:.4f}, F1: {multitask_results['sentiment_f1_macro']:.4f}")
    print(f"  Emotion - Accuracy: {multitask_results['emotion_accuracy']:.4f}, F1: {multitask_results['emotion_f1_macro']:.4f}")
    print(f"  Combined - Accuracy: {multitask_results['combined_accuracy']:.4f}, F1: {multitask_results['combined_f1_macro']:.4f}")
    
    print(f"\n📈 BERTWEET COMPARISON:")
    print(f"  Single-task Sentiment vs Multi-task Sentiment:")
    print(f"    Accuracy: {sentiment_results['accuracy']:.4f} vs {multitask_results['sentiment_accuracy']:.4f}")
    print(f"    F1 Macro: {sentiment_results['f1_macro']:.4f} vs {multitask_results['sentiment_f1_macro']:.4f}")
    
    print(f"  Single-task Emotion vs Multi-task Emotion:")
    print(f"    Accuracy: {emotion_results['accuracy']:.4f} vs {multitask_results['emotion_accuracy']:.4f}")
    print(f"    F1 Macro: {emotion_results['f1_macro']:.4f} vs {multitask_results['emotion_f1_macro']:.4f}")
    
    print("="*80)

print("✅ BERTweet Evaluation functions defined!")


# In[ ]:


def main_bertweet_training_pipeline():
    
    print("STARTING COMPREHENSIVE BERTWEET TRAINING PIPELINE")
    print("="*80)
    
    # Load and process datasets for BERTweet
    print("\n1️⃣ Loading and processing datasets for BERTweet...")
    sentiment_data, emotion_data = load_and_process_datasets_bertweet()
    multitask_data = create_multitask_data_bertweet(sentiment_data, emotion_data)
    
    # Load Reddit evaluation data
    print("\nLoading Reddit evaluation data...")
    reddit_data = load_reddit_evaluation_data()
    
    # Model configurations
    model_name = "vinai/bertweet-base"
    n_trials = 10 # Number of hyperparameter tuning trials
    
    # Store results globally
    all_results = {}
    
    print("✅ Data loading completed!")
    print(f"📊 Sentiment data: {len(sentiment_data['train']['texts'])} train samples")
    print(f"📊 Emotion data: {len(emotion_data['train']['texts'])} train samples")
    print(f"📊 Multitask data: {len(multitask_data['train']['texts'])} train samples")
    if reddit_data:
        print(f"📊 Reddit data: {len(reddit_data['sentiment']['texts'])} evaluation samples")
    
    # ==============================================================================
    # PHASE 1: INITIAL BERTWEET TRAINING WITH DEFAULT PARAMETERS
    # ==============================================================================
    print(f"\n" + "="*80)
    print(f"📍 PHASE 1: INITIAL BERTWEET TRAINING WITH DEFAULT PARAMETERS")
    print(f"="*80)
    
    # Default configuration for BERTweet
    default_config_sentiment = TrainingConfig(
        model_name=model_name,
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        max_length=128,
        task_type="sentiment",
        output_dir="./initial_bertweet_sentiment_model"
    )
    
    default_config_emotion = TrainingConfig(
        model_name=model_name,
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        max_length=128,
        task_type="emotion",
        output_dir="./initial_bertweet_emotion_model"
    )
    
    default_config_multitask = TrainingConfig(
        model_name=model_name,
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        max_length=128,
        alpha=0.5,
        task_type="multitask",
        output_dir="./initial_bertweet_multitask_model"
    )
    
    # 1.1 Train Initial BERTweet Sentiment Model
    print(f"\n2️⃣ Training Initial BERTweet Sentiment Model...")
    print("="*60)
    
    initial_sentiment_trainer = BERTweetSingleTaskTrainer(
        config=default_config_sentiment,
        num_classes=bertweet_model_config.sentiment_num_classes
    )
    initial_sentiment_history = initial_sentiment_trainer.train(sentiment_data)
    
    # Evaluate initial BERTweet sentiment model on both general and Reddit datasets
    initial_sentiment_results = evaluate_bertweet_model(
        model_path="./initial_bertweet_sentiment_model/model_best",
        model_type="sentiment",
        test_data=sentiment_data['test'],
        model_name=model_name,
        reddit_data=reddit_data['sentiment'] if reddit_data else None
    )
    all_results['initial_sentiment'] = initial_sentiment_results
    
    # 1.2 Train Initial BERTweet Emotion Model
    print(f"\n3️⃣ Training Initial BERTweet Emotion Model...")
    print("="*60)
    
    initial_emotion_trainer = BERTweetSingleTaskTrainer(
        config=default_config_emotion,
        num_classes=bertweet_model_config.emotion_num_classes
    )
    initial_emotion_history = initial_emotion_trainer.train(emotion_data)
    
    # Evaluate initial BERTweet emotion model on both general and Reddit datasets
    initial_emotion_results = evaluate_bertweet_model(
        model_path="./initial_bertweet_emotion_model/model_best",
        model_type="emotion",
        test_data=emotion_data['test'],
        model_name=model_name,
        reddit_data=reddit_data['emotion'] if reddit_data else None
    )
    all_results['initial_emotion'] = initial_emotion_results
    
    # 1.3 Train Initial BERTweet Multi-task Model
    print(f"\n4️⃣ Training Initial BERTweet Multi-task Model...")
    print("="*60)
    
    initial_multitask_trainer = BERTweetMultiTaskTrainer(config=default_config_multitask)
    initial_multitask_history = initial_multitask_trainer.train(multitask_data)
    
    # Evaluate initial multitask model on both general and Reddit datasets
    initial_multitask_results = evaluate_bertweet_model(
        model_path="./initial_bertweet_multitask_model/model_best",
        model_type="multitask",
        test_data=multitask_data['test'],
        model_name=model_name,
        reddit_data=reddit_data['multitask'] if reddit_data else None
    )
    all_results['initial_multitask'] = initial_multitask_results
    
    # Display initial BERTweet results summary
    print(f"\n5️⃣ Initial BERTweet Results Summary...")
    print("="*60)
    create_bertweet_initial_results_summary(
        sentiment_results=all_results['initial_sentiment'],
        emotion_results=all_results['initial_emotion'],
        multitask_results=all_results['initial_multitask']
    )
    
    # ==============================================================================
    # PHASE 2: BERTWEET HYPERPARAMETER TUNING
    # ==============================================================================
    print(f"\n" + "="*80)
    print(f"📍 PHASE 2: BERTWEET HYPERPARAMETER TUNING")
    print(f"="*80)
    
    # 2.1 Hyperparameter tuning for BERTweet sentiment
    print(f"\n6️⃣ Hyperparameter Tuning for BERTweet Sentiment Model...")
    print("="*60)
    
    sentiment_tuner = BERTweetHyperparameterTuner(
        model_type="sentiment",
        data_splits=sentiment_data,
        n_trials=n_trials,
        model_name=model_name
    )
    sentiment_study = sentiment_tuner.tune()
    
    # 2.2 Hyperparameter tuning for BERTweet emotion
    print(f"\n7️⃣ Hyperparameter Tuning for BERTweet Emotion Model...")
    print("="*60)
    
    emotion_tuner = BERTweetHyperparameterTuner(
        model_type="emotion",
        data_splits=emotion_data,
        n_trials=n_trials,
        model_name=model_name
    )
    emotion_study = emotion_tuner.tune()
    
    # 2.3 Hyperparameter tuning for BERTweet multi-task
    print(f"\n8️⃣ Hyperparameter Tuning for BERTweet Multi-task Model...")
    print("="*60)
    
    multitask_tuner = BERTweetHyperparameterTuner(
        model_type="multitask",
        data_splits=multitask_data,
        n_trials=n_trials,
        model_name=model_name
    )
    multitask_study = multitask_tuner.tune()
    
    # ==============================================================================
    # PHASE 3: FINAL BERTWEET TRAINING WITH OPTIMIZED PARAMETERS
    # ==============================================================================
    print(f"\n" + "="*80)
    print(f"📍 PHASE 3: FINAL BERTWEET TRAINING WITH OPTIMIZED PARAMETERS")
    print(f"="*80)
    
    # 3.1 Train optimized BERTweet sentiment model
    print(f"\n9️⃣ Training Optimized BERTweet Sentiment Model...")
    print("="*60)
    
    optimized_sentiment_trainer, optimized_sentiment_history = train_bertweet_with_best_params(
        model_type="sentiment",
        data_splits=sentiment_data,
        best_params=sentiment_study.best_params,
        model_name=model_name
    )
    
    # Evaluate optimized BERTweet sentiment model
    optimized_sentiment_results = evaluate_bertweet_model(
        model_path="./final_bertweet_sentiment_model/model_best",
        model_type="sentiment",
        test_data=sentiment_data['test'],
        model_name=model_name,
        reddit_data=reddit_data['sentiment'] if reddit_data else None
    )
    all_results['optimized_sentiment'] = optimized_sentiment_results
    
    # 3.2 Train optimized BERTweet emotion model
    print(f"\n🔟 Training Optimized BERTweet Emotion Model...")
    print("="*60)
    
    optimized_emotion_trainer, optimized_emotion_history = train_bertweet_with_best_params(
        model_type="emotion",
        data_splits=emotion_data,
        best_params=emotion_study.best_params,
        model_name=model_name
    )
    
    # Evaluate optimized BERTweet emotion model
    optimized_emotion_results = evaluate_bertweet_model(
        model_path="./final_bertweet_emotion_model/model_best",
        model_type="emotion",
        test_data=emotion_data['test'],
        model_name=model_name,
        reddit_data=reddit_data['emotion'] if reddit_data else None
    )
    all_results['optimized_emotion'] = optimized_emotion_results
    
    # 3.3 Train optimized BERTweet multi-task model
    print(f"\n1️⃣1️⃣ Training Optimized BERTweet Multi-task Model...")
    print("="*60)
    
    optimized_multitask_trainer, optimized_multitask_history = train_bertweet_with_best_params(
        model_type="multitask",
        data_splits=multitask_data,
        best_params=multitask_study.best_params,
        model_name=model_name
    )
    
    # Evaluate optimized BERTweet multi-task model
    optimized_multitask_results = evaluate_bertweet_model(
        model_path="./final_bertweet_multitask_model/model_best",
        model_type="multitask",
        test_data=multitask_data['test'],
        model_name=model_name,
        reddit_data=reddit_data['multitask'] if reddit_data else None
    )
    all_results['optimized_multitask'] = optimized_multitask_results
    
    # ==============================================================================
    # PHASE 4: COMPREHENSIVE BERTWEET RESULTS COMPARISON
    # ==============================================================================
    print(f"\n" + "="*80)
    print(f"📍 PHASE 4: COMPREHENSIVE BERTWEET RESULTS COMPARISON")
    print(f"="*80)
    
    # Create comprehensive BERTweet comparison
    create_comprehensive_bertweet_results_comparison(all_results)
    
    # Save all BERTweet results
    results_summary = {
        'model_type': 'BERTweet',
        'model_name': model_name,
        'initial_models': {
            'sentiment': all_results['initial_sentiment'],
            'emotion': all_results['initial_emotion'],
            'multitask': all_results['initial_multitask']
        },
        'optimized_models': {
            'sentiment': all_results['optimized_sentiment'],
            'emotion': all_results['optimized_emotion'],
            'multitask': all_results['optimized_multitask']
        },
        'hyperparameter_studies': {
            'sentiment': sentiment_study.best_params,
            'emotion': emotion_study.best_params,
            'multitask': multitask_study.best_params
        },
        'improvements': {
            'sentiment': {
                'accuracy_improvement': all_results['optimized_sentiment']['accuracy'] - all_results['initial_sentiment']['accuracy'],
                'f1_improvement': all_results['optimized_sentiment']['f1_macro'] - all_results['initial_sentiment']['f1_macro']
            },
            'emotion': {
                'accuracy_improvement': all_results['optimized_emotion']['accuracy'] - all_results['initial_emotion']['accuracy'],
                'f1_improvement': all_results['optimized_emotion']['f1_macro'] - all_results['initial_emotion']['f1_macro']
            },
            'multitask': {
                'sentiment_accuracy_improvement': all_results['optimized_multitask']['sentiment_accuracy'] - all_results['initial_multitask']['sentiment_accuracy'],
                'emotion_accuracy_improvement': all_results['optimized_multitask']['emotion_accuracy'] - all_results['initial_multitask']['emotion_accuracy'],
                'combined_f1_improvement': all_results['optimized_multitask']['combined_f1_macro'] - all_results['initial_multitask']['combined_f1_macro']
            }
        }
    }
    
    with open('comprehensive_bertweet_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ COMPLETE BERTWEET PIPELINE FINISHED!")
    print(f"📁 Results saved to: comprehensive_bertweet_results_summary.json")
    print(f"📁 Initial models saved to: ./initial_bertweet_*_model/")
    print(f"📁 Optimized models saved to: ./final_bertweet_*_model/")
    
    return all_results

def create_bertweet_initial_results_summary(sentiment_results: Dict, emotion_results: Dict, multitask_results: Dict):
    
    print(f"\n📊 INITIAL BERTWEET MODELS RESULTS SUMMARY")
    print(f"="*60)
    
    print(f"\n🎯 INITIAL BERTWEET SENTIMENT MODEL:")
    print(f"  Accuracy: {sentiment_results['accuracy']:.4f}")
    print(f"  F1 Macro: {sentiment_results['f1_macro']:.4f}")
    
    print(f"\n😊 INITIAL BERTWEET EMOTION MODEL:")
    print(f"  Accuracy: {emotion_results['accuracy']:.4f}")
    print(f"  F1 Macro: {emotion_results['f1_macro']:.4f}")
    
    print(f"\n🔗 INITIAL BERTWEET MULTI-TASK MODEL:")
    print(f"  General Dataset:")
    print(f"    Sentiment - Accuracy: {multitask_results['general']['sentiment_accuracy']:.4f}, F1: {multitask_results['general']['sentiment_f1_macro']:.4f}")
    print(f"    Emotion - Accuracy: {multitask_results['general']['emotion_accuracy']:.4f}, F1: {multitask_results['general']['emotion_f1_macro']:.4f}")
    print(f"    Combined - Accuracy: {multitask_results['general']['combined_accuracy']:.4f}, F1: {multitask_results['general']['combined_f1_macro']:.4f}")
    if multitask_results.get('reddit'):
        print(f"  Reddit Dataset:")
        print(f"    Sentiment - Accuracy: {multitask_results['reddit']['sentiment_accuracy']:.4f}, F1: {multitask_results['reddit']['sentiment_f1_macro']:.4f}")
        print(f"    Emotion - Accuracy: {multitask_results['reddit']['emotion_accuracy']:.4f}, F1: {multitask_results['reddit']['emotion_f1_macro']:.4f}")
        print(f"    Combined - Accuracy: {multitask_results['reddit']['combined_accuracy']:.4f}, F1: {multitask_results['reddit']['combined_f1_macro']:.4f}")
    
    print(f"\n�� These are BERTweet baseline results. Hyperparameter tuning will aim to improve them!")

def create_comprehensive_bertweet_results_comparison(all_results: Dict):
    
    print(f"\n📊 COMPREHENSIVE BERTWEET RESULTS COMPARISON")
    print(f"="*80)
    
    print(f"\n🎯 BERTWEET SENTIMENT MODEL COMPARISON:")
    print(f"  Initial    - Accuracy: {all_results['initial_sentiment']['accuracy']:.4f}, F1: {all_results['initial_sentiment']['f1_macro']:.4f}")
    print(f"  Optimized  - Accuracy: {all_results['optimized_sentiment']['accuracy']:.4f}, F1: {all_results['optimized_sentiment']['f1_macro']:.4f}")
    
    sent_acc_improve = all_results['optimized_sentiment']['accuracy'] - all_results['initial_sentiment']['accuracy']
    sent_f1_improve = all_results['optimized_sentiment']['f1_macro'] - all_results['initial_sentiment']['f1_macro']
    print(f"  Improvement - Accuracy: {sent_acc_improve:+.4f}, F1: {sent_f1_improve:+.4f}")
    
    print(f"\n😊 BERTWEET EMOTION MODEL COMPARISON:")
    print(f"  Initial    - Accuracy: {all_results['initial_emotion']['accuracy']:.4f}, F1: {all_results['initial_emotion']['f1_macro']:.4f}")
    print(f"  Optimized  - Accuracy: {all_results['optimized_emotion']['accuracy']:.4f}, F1: {all_results['optimized_emotion']['f1_macro']:.4f}")
    
    emo_acc_improve = all_results['optimized_emotion']['accuracy'] - all_results['initial_emotion']['accuracy']
    emo_f1_improve = all_results['optimized_emotion']['f1_macro'] - all_results['initial_emotion']['f1_macro']
    print(f"  Improvement - Accuracy: {emo_acc_improve:+.4f}, F1: {emo_f1_improve:+.4f}")
    
    print(f"\n🔗 BERTWEET MULTI-TASK MODEL COMPARISON:")
    print(f"  SENTIMENT TASK:")
    print(f"    Initial    - Accuracy: {all_results['initial_multitask']['sentiment_accuracy']:.4f}, F1: {all_results['initial_multitask']['sentiment_f1_macro']:.4f}")
    print(f"    Optimized  - Accuracy: {all_results['optimized_multitask']['sentiment_accuracy']:.4f}, F1: {all_results['optimized_multitask']['sentiment_f1_macro']:.4f}")
    
    mt_sent_acc_improve = all_results['optimized_multitask']['sentiment_accuracy'] - all_results['initial_multitask']['sentiment_accuracy']
    mt_sent_f1_improve = all_results['optimized_multitask']['sentiment_f1_macro'] - all_results['initial_multitask']['sentiment_f1_macro']
    print(f"    Improvement - Accuracy: {mt_sent_acc_improve:+.4f}, F1: {mt_sent_f1_improve:+.4f}")
    
    print(f"  EMOTION TASK:")
    print(f"    Initial    - Accuracy: {all_results['initial_multitask']['emotion_accuracy']:.4f}, F1: {all_results['initial_multitask']['emotion_f1_macro']:.4f}")
    print(f"    Optimized  - Accuracy: {all_results['optimized_multitask']['emotion_accuracy']:.4f}, F1: {all_results['optimized_multitask']['emotion_f1_macro']:.4f}")
    
    mt_emo_acc_improve = all_results['optimized_multitask']['emotion_accuracy'] - all_results['initial_multitask']['emotion_accuracy']
    mt_emo_f1_improve = all_results['optimized_multitask']['emotion_f1_macro'] - all_results['initial_multitask']['emotion_f1_macro']
    print(f"    Improvement - Accuracy: {mt_emo_acc_improve:+.4f}, F1: {mt_emo_f1_improve:+.4f}")
    
    print(f"  COMBINED:")
    print(f"    Initial    - Accuracy: {all_results['initial_multitask']['combined_accuracy']:.4f}, F1: {all_results['initial_multitask']['combined_f1_macro']:.4f}")
    print(f"    Optimized  - Accuracy: {all_results['optimized_multitask']['combined_accuracy']:.4f}, F1: {all_results['optimized_multitask']['combined_f1_macro']:.4f}")
    
    mt_combined_acc_improve = all_results['optimized_multitask']['combined_accuracy'] - all_results['initial_multitask']['combined_accuracy']
    mt_combined_f1_improve = all_results['optimized_multitask']['combined_f1_macro'] - all_results['initial_multitask']['combined_f1_macro']
    print(f"    Improvement - Accuracy: {mt_combined_acc_improve:+.4f}, F1: {mt_combined_f1_improve:+.4f}")
    
    print(f"\n📈 BERTWEET SINGLE-TASK vs MULTI-TASK COMPARISON (OPTIMIZED):")
    print(f"  SENTIMENT:")
    print(f"    Single-task: Accuracy: {all_results['optimized_sentiment']['accuracy']:.4f}, F1: {all_results['optimized_sentiment']['f1_macro']:.4f}")
    print(f"    Multi-task:  Accuracy: {all_results['optimized_multitask']['sentiment_accuracy']:.4f}, F1: {all_results['optimized_multitask']['sentiment_f1_macro']:.4f}")
    
    print(f"  EMOTION:")
    print(f"    Single-task: Accuracy: {all_results['optimized_emotion']['accuracy']:.4f}, F1: {all_results['optimized_emotion']['f1_macro']:.4f}")
    print(f"    Multi-task:  Accuracy: {all_results['optimized_multitask']['emotion_accuracy']:.4f}, F1: {all_results['optimized_multitask']['emotion_f1_macro']:.4f}")
    
    print("="*80)

print("BERTweet Main training pipeline defined!")


# In[ ]:


print("🚀 STARTING BERTWEET TRAINING PIPELINE")
print("=" * 80)

# Clear memory before starting
aggressive_memory_cleanup()

# Load and process datasets for BERTweet
print("\n1️⃣ Loading and processing datasets for BERTweet...")
sentiment_data, emotion_data = load_and_process_datasets_bertweet()
multitask_data = create_multitask_data_bertweet(sentiment_data, emotion_data)

# Load Reddit evaluation data
print("\nLoading Reddit evaluation data...")
reddit_data = load_reddit_evaluation_data()

# Model configurations
model_name = "vinai/bertweet-base"
n_trials = 10 # Number of hyperparameter tuning trials

# Store results globally
all_results = {}

print("✅ Data loading completed!")
print(f"📊 Sentiment data: {len(sentiment_data['train']['texts'])} train samples")
print(f"📊 Emotion data: {len(emotion_data['train']['texts'])} train samples")
print(f"📊 Multitask data: {len(multitask_data['train']['texts'])} train samples")
if reddit_data:
    print(f"📊 Reddit data: {len(reddit_data['sentiment']['texts'])} evaluation samples")


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 1: INITIAL BERTWEET TRAINING - SENTIMENT MODEL")
print("="*80)

# Default configuration for BERTweet sentiment
default_config_sentiment = TrainingConfig(
    model_name=model_name,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    max_length=128,
    task_type="sentiment",
    output_dir="./initial_bertweet_sentiment_model"
)

print("\n2️⃣ Training Initial BERTweet Sentiment Model...")
print("="*60)

# Train initial sentiment model
initial_sentiment_trainer = BERTweetSingleTaskTrainer(
    config=default_config_sentiment,
    num_classes=bertweet_model_config.sentiment_num_classes
)
initial_sentiment_history = initial_sentiment_trainer.train(sentiment_data)

# Evaluate initial sentiment model on both general and Reddit datasets
initial_sentiment_results = evaluate_bertweet_model(
    model_path="./initial_bertweet_sentiment_model/model_best",
    model_type="sentiment",
    test_data=sentiment_data['test'],
    model_name=model_name,
    reddit_data=reddit_data['sentiment'] if reddit_data else None
)
all_results['initial_sentiment'] = initial_sentiment_results

print(f"\n✅ Initial Sentiment Model Results:")
print(f"  General Dataset:")
print(f"    Accuracy: {initial_sentiment_results['general']['accuracy']:.4f}")
print(f"    F1 Macro: {initial_sentiment_results['general']['f1_macro']:.4f}")
if initial_sentiment_results.get('reddit'):
    print(f"  Reddit Dataset:")
    print(f"    Accuracy: {initial_sentiment_results['reddit']['accuracy']:.4f}")
    print(f"    F1 Macro: {initial_sentiment_results['reddit']['f1_macro']:.4f}")

# Clean up memory
aggressive_memory_cleanup()


# In[ ]:


# Cell 3: Initial Emotion Model Training
print("\n" + "="*80)
print("📍 PHASE 1: INITIAL BERTWEET TRAINING - EMOTION MODEL")
print("="*80)

# Default configuration for BERTweet emotion
default_config_emotion = TrainingConfig(
    model_name=model_name,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    max_length=128,
    task_type="emotion",
    output_dir="./initial_bertweet_emotion_model"
)

print("\nTraining Initial BERTweet Emotion Model...")
print("="*60)

# Train initial emotion model
initial_emotion_trainer = BERTweetSingleTaskTrainer(
    config=default_config_emotion,
    num_classes=bertweet_model_config.emotion_num_classes
)
initial_emotion_history = initial_emotion_trainer.train(emotion_data)

# Evaluate initial emotion model on both general and Reddit datasets
initial_emotion_results = evaluate_bertweet_model(
    model_path="./initial_bertweet_emotion_model/model_best",
    model_type="emotion",
    test_data=emotion_data['test'],
    model_name=model_name,
    reddit_data=reddit_data['emotion'] if reddit_data else None
)
all_results['initial_emotion'] = initial_emotion_results

print(f"\n✅ Initial Emotion Model Results:")
print(f"  General Dataset:")
print(f"    Accuracy: {initial_emotion_results['general']['accuracy']:.4f}")
print(f"    F1 Macro: {initial_emotion_results['general']['f1_macro']:.4f}")
if initial_emotion_results.get('reddit'):
    print(f"  Reddit Dataset:")
    print(f"    Accuracy: {initial_emotion_results['reddit']['accuracy']:.4f}")
    print(f"    F1 Macro: {initial_emotion_results['reddit']['f1_macro']:.4f}")

# Clean up memory
aggressive_memory_cleanup()


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 1: INITIAL BERTWEET TRAINING - MULTITASK MODEL")
print("="*80)

# Default configuration for BERTweet multitask
default_config_multitask = TrainingConfig(
    model_name=model_name,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    max_length=128,
    alpha=0.5,
    task_type="multitask",
    output_dir="./initial_bertweet_multitask_model"
)

print("\n4️⃣ Training Initial BERTweet Multi-task Model...")
print("="*60)

# Train initial multitask model
initial_multitask_trainer = BERTweetMultiTaskTrainer(config=default_config_multitask)
initial_multitask_history = initial_multitask_trainer.train(multitask_data)

# Evaluate initial multitask model on both general and Reddit datasets
initial_multitask_results = evaluate_bertweet_model(
    model_path="./initial_bertweet_multitask_model/model_best",
    model_type="multitask",
    test_data=multitask_data['test'],
    model_name=model_name,
    reddit_data=reddit_data['multitask'] if reddit_data else None
)
all_results['initial_multitask'] = initial_multitask_results

print(f"\n✅ Initial Multitask Model Results:")
print(f"  General Dataset:")
print(f"    Sentiment - Accuracy: {initial_multitask_results['general']['sentiment_accuracy']:.4f}, F1: {initial_multitask_results['general']['sentiment_f1_macro']:.4f}")
print(f"    Emotion - Accuracy: {initial_multitask_results['general']['emotion_accuracy']:.4f}, F1: {initial_multitask_results['general']['emotion_f1_macro']:.4f}")
print(f"    Combined - Accuracy: {initial_multitask_results['general']['combined_accuracy']:.4f}, F1: {initial_multitask_results['general']['combined_f1_macro']:.4f}")
if initial_multitask_results.get('reddit'):
    print(f"  Reddit Dataset:")
    print(f"    Sentiment - Accuracy: {initial_multitask_results['reddit']['sentiment_accuracy']:.4f}, F1: {initial_multitask_results['reddit']['sentiment_f1_macro']:.4f}")
    print(f"    Emotion - Accuracy: {initial_multitask_results['reddit']['emotion_accuracy']:.4f}, F1: {initial_multitask_results['reddit']['emotion_f1_macro']:.4f}")
    print(f"    Combined - Accuracy: {initial_multitask_results['reddit']['combined_accuracy']:.4f}, F1: {initial_multitask_results['reddit']['combined_f1_macro']:.4f}")

# Clean up memory
aggressive_memory_cleanup()


# In[ ]:


print("\n" + "="*80)
print("📍 INITIAL BERTWEET RESULTS SUMMARY")
print("="*80)

create_bertweet_initial_results_summary(
    sentiment_results=all_results['initial_sentiment'],
    emotion_results=all_results['initial_emotion'],
    multitask_results=all_results['initial_multitask']
)

print(f"\n💡 These are BERTweet baseline results. Now proceeding to hyperparameter tuning!")


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 2: ULTRA-FAST HYPERPARAMETER TUNING - SENTIMENT")
print("="*80)

print("\n6️⃣ Fast Hyperparameter Tuning for BERTweet Sentiment Model...")
print("="*60)

# Create FAST tuner for sentiment
sentiment_tuner = FastBERTweetHyperparameterTuner(
    model_type="sentiment",
    data_splits=sentiment_data,
    n_trials=5,  # Even fewer trials for speed
    model_name=model_name,
    subset_ratio=0.02,  # Only 2% of data!
    max_epochs_per_trial=2  # Only 2 epochs per trial!
)

# Run hyperparameter tuning
sentiment_study = sentiment_tuner.tune()

print(f"\n✅ Sentiment Hyperparameter Tuning Completed!")
print(f"🏆 Best F1 Score: {sentiment_study.best_value:.4f}")
print(f"📋 Best Parameters:")
for key, value in sentiment_study.best_params.items():
    print(f"  {key}: {value}")

# Clean up memory
aggressive_memory_cleanup()


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 2: HYPERPARAMETER TUNING - EMOTION")
print("="*80)

print("\n7️⃣ Hyperparameter Tuning for BERTweet Emotion Model...")
print("="*60)

# Create tuner for emotion
emotion_tuner = FastBERTweetHyperparameterTuner(
    model_type="emotion",
    data_splits=emotion_data,
    n_trials=5,  # Even fewer trials for speed
    model_name=model_name,
    subset_ratio=0.02,  # Only 2% of data!
    max_epochs_per_trial=2  # Only 2 epochs per trial!
)

# Run hyperparameter tuning
emotion_study = emotion_tuner.tune()

print(f"\n✅ Emotion Hyperparameter Tuning Completed!")
print(f"🏆 Best F1 Score: {emotion_study.best_value:.4f}")
print(f"📋 Best Parameters:")
for key, value in emotion_study.best_params.items():
    print(f"  {key}: {value}")

# Clean up memory
aggressive_memory_cleanup()


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 2: ULTRA-FAST HYPERPARAMETER TUNING - MULTITASK")
print("="*80)

print("\n8️⃣ Fast Hyperparameter Tuning for BERTweet Multi-task Model...")
print("="*60)

multitask_tuner = FastBERTweetHyperparameterTuner(
    model_type="multitask",
    data_splits=multitask_data,
    n_trials=5,  # Reduced trials for speed
    model_name=model_name,
    subset_ratio=0.02,  # Only 2% of data!
    max_epochs_per_trial=2  # Only 2 epochs per trial!
)

multitask_study = multitask_tuner.tune()

print(f"\n✅ Multitask Hyperparameter Tuning Completed!")
print(f"🏆 Best Combined F1 Score: {multitask_study.best_value:.4f}")
print(f"📋 Best Parameters:")
for key, value in multitask_study.best_params.items():
    print(f"  {key}: {value}")

# Clean up memory
aggressive_memory_cleanup()


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 3: FINAL TRAINING - OPTIMIZED SENTIMENT MODEL")
print("="*80)

print("\n9️⃣ Training Final BERTweet Sentiment Model with Best Parameters...")
print("="*60)

# Get best parameters from sentiment tuning
best_sentiment_params = sentiment_study.best_params
print(f"🎯 Using best hyperparameters:")
for key, value in best_sentiment_params.items():
    print(f"  {key}: {value}")

# Create optimized config for final training (full dataset, more epochs)
final_sentiment_config = TrainingConfig(
    model_name=model_name,
    learning_rate=best_sentiment_params['learning_rate'],
    batch_size=best_sentiment_params['batch_size'],
    num_epochs=5,  # Increase epochs for final training
    warmup_ratio=best_sentiment_params['warmup_ratio'],
    weight_decay=best_sentiment_params['weight_decay'],
    hidden_dropout_prob=best_sentiment_params['hidden_dropout_prob'],
    classifier_dropout=best_sentiment_params['classifier_dropout'],
    max_length=best_sentiment_params.get('max_length', 128),  # Fixed: use .get() with default
    task_type="sentiment",
    output_dir="./final_bertweet_sentiment_model"
)

print(f"\n🚀 Training final sentiment model:")
print(f"  Dataset: Full sentiment data ({len(sentiment_data['train']['texts'])} train samples)")
print(f"  Epochs: {final_sentiment_config.num_epochs}")
print(f"  Batch size: {final_sentiment_config.batch_size}")
print(f"  Learning rate: {final_sentiment_config.learning_rate:.2e}")
print(f"  Max length: {final_sentiment_config.max_length}")

# Train final sentiment model
final_sentiment_trainer = BERTweetSingleTaskTrainer(
    config=final_sentiment_config,
    num_classes=bertweet_model_config.sentiment_num_classes
)
final_sentiment_history = final_sentiment_trainer.train(sentiment_data)

# Evaluate final sentiment model on both general and Reddit datasets
final_sentiment_results = evaluate_bertweet_model(
    model_path="./final_bertweet_sentiment_model/model_best",
    model_type="sentiment",
    test_data=sentiment_data['test'],
    model_name=model_name,
    reddit_data=reddit_data['sentiment'] if reddit_data else None
)

print(f"\n✅ Final Sentiment Model Results:")
print(f"  General Dataset:")
print(f"    Accuracy: {final_sentiment_results['general']['accuracy']:.4f}")
print(f"    F1 Macro: {final_sentiment_results['general']['f1_macro']:.4f}")
if final_sentiment_results.get('reddit'):
    print(f"  Reddit Dataset:")
    print(f"    Accuracy: {final_sentiment_results['reddit']['accuracy']:.4f}")
    print(f"    F1 Macro: {final_sentiment_results['reddit']['f1_macro']:.4f}")

# Compare with tuning results
print(f"\n📊 Comparison:")
print(f"  Tuning F1 (on subset): {sentiment_study.best_value:.4f}")
print(f"  Final F1 (on full test): {final_sentiment_results['f1_macro']:.4f}")

# Clean up memory
aggressive_memory_cleanup()
print(f"💾 Final sentiment model saved to: ./final_bertweet_sentiment_model/")


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 3: FINAL TRAINING - OPTIMIZED EMOTION MODEL")
print("="*80)

print("\n🔟 Training Final BERTweet Emotion Model with Best Parameters...")
print("="*60)

# Get best parameters from emotion tuning
best_emotion_params = emotion_study.best_params
print(f"🎯 Using best hyperparameters:")
for key, value in best_emotion_params.items():
    print(f"  {key}: {value}")

# Create optimized config for final training (full dataset, more epochs)
final_emotion_config = TrainingConfig(
    model_name=model_name,
    learning_rate=best_emotion_params['learning_rate'],
    batch_size=best_emotion_params['batch_size'],
    num_epochs=5,  # Increase epochs for final training
    warmup_ratio=best_emotion_params['warmup_ratio'],
    weight_decay=best_emotion_params['weight_decay'],
    hidden_dropout_prob=best_emotion_params['hidden_dropout_prob'],
    classifier_dropout=best_emotion_params['classifier_dropout'],
    max_length=best_emotion_params.get('max_length', 128),  # Fixed: use .get() with default
    task_type="emotion",
    output_dir="./final_bertweet_emotion_model"
)

print(f"\n🚀 Training final emotion model:")
print(f"  Dataset: Full emotion data ({len(emotion_data['train']['texts'])} train samples)")
print(f"  Epochs: {final_emotion_config.num_epochs}")
print(f"  Batch size: {final_emotion_config.batch_size}")
print(f"  Learning rate: {final_emotion_config.learning_rate:.2e}")
print(f"  Max length: {final_emotion_config.max_length}")

# Train final emotion model
final_emotion_trainer = BERTweetSingleTaskTrainer(
    config=final_emotion_config,
    num_classes=bertweet_model_config.emotion_num_classes
)
final_emotion_history = final_emotion_trainer.train(emotion_data)

# Evaluate final emotion model on both general and Reddit datasets
final_emotion_results = evaluate_bertweet_model(
    model_path="./final_bertweet_emotion_model/model_best",
    model_type="emotion",
    test_data=emotion_data['test'],
    model_name=model_name,
    reddit_data=reddit_data['emotion'] if reddit_data else None
)

print(f"\n✅ Final Emotion Model Results:")
print(f"  General Dataset:")
print(f"    Accuracy: {final_emotion_results['general']['accuracy']:.4f}")
print(f"    F1 Macro: {final_emotion_results['general']['f1_macro']:.4f}")
if final_emotion_results.get('reddit'):
    print(f"  Reddit Dataset:")
    print(f"    Accuracy: {final_emotion_results['reddit']['accuracy']:.4f}")
    print(f"    F1 Macro: {final_emotion_results['reddit']['f1_macro']:.4f}")

# Compare with tuning results
print(f"\n📊 Comparison:")
print(f"  Tuning F1 (on subset): {emotion_study.best_value:.4f}")
print(f"  Final F1 (on full test): {final_emotion_results['f1_macro']:.4f}")

# Clean up memory
aggressive_memory_cleanup()
print(f"💾 Final emotion model saved to: ./final_bertweet_emotion_model/")


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 3: FINAL TRAINING - OPTIMIZED MULTITASK MODEL")
print("="*80)

print("\n1️⃣1️⃣ Training Final BERTweet Multi-task Model with Best Parameters...")
print("="*60)

# Get best parameters from multitask tuning
best_multitask_params = multitask_study.best_params
print(f"🎯 Using best hyperparameters:")
for key, value in best_multitask_params.items():
    print(f"  {key}: {value}")

# Create optimized config for final training (full dataset, more epochs)
final_multitask_config = TrainingConfig(
    model_name=model_name,
    learning_rate=best_multitask_params['learning_rate'],
    batch_size=best_multitask_params['batch_size'],
    num_epochs=5,  # Increase epochs for final training
    warmup_ratio=best_multitask_params['warmup_ratio'],
    weight_decay=best_multitask_params['weight_decay'],
    hidden_dropout_prob=best_multitask_params['hidden_dropout_prob'],
    classifier_dropout=best_multitask_params['classifier_dropout'],
    max_length=best_multitask_params.get('max_length', 128),  # Fixed: use .get() with default
    alpha=best_multitask_params['alpha'],  # Multitask-specific parameter
    task_type="multitask",
    output_dir="./final_bertweet_multitask_model"
)

print(f"\n🚀 Training final multitask model:")
print(f"  Dataset: Full multitask data ({len(multitask_data['train']['texts'])} train samples)")
print(f"  Epochs: {final_multitask_config.num_epochs}")
print(f"  Batch size: {final_multitask_config.batch_size}")
print(f"  Learning rate: {final_multitask_config.learning_rate:.2e}")
print(f"  Max length: {final_multitask_config.max_length}")
print(f"  Alpha (loss weighting): {final_multitask_config.alpha:.3f}")

# Train final multitask model
final_multitask_trainer = BERTweetMultiTaskTrainer(config=final_multitask_config)
final_multitask_history = final_multitask_trainer.train(multitask_data)

# Evaluate final multitask model on both general and Reddit datasets
final_multitask_results = evaluate_bertweet_model(
    model_path="./final_bertweet_multitask_model/model_best",
    model_type="multitask",
    test_data=multitask_data['test'],
    model_name=model_name,
    reddit_data=reddit_data['multitask'] if reddit_data else None
)

print(f"\n✅ Final Multitask Model Results:")
print(f"  General Dataset:")
print(f"    Sentiment - Accuracy: {final_multitask_results['general']['sentiment_accuracy']:.4f}, F1: {final_multitask_results['general']['sentiment_f1_macro']:.4f}")
print(f"    Emotion - Accuracy: {final_multitask_results['general']['emotion_accuracy']:.4f}, F1: {final_multitask_results['general']['emotion_f1_macro']:.4f}")
print(f"    Combined - Accuracy: {final_multitask_results['general']['combined_accuracy']:.4f}, F1: {final_multitask_results['general']['combined_f1_macro']:.4f}")
if final_multitask_results.get('reddit'):
    print(f"  Reddit Dataset:")
    print(f"    Sentiment - Accuracy: {final_multitask_results['reddit']['sentiment_accuracy']:.4f}, F1: {final_multitask_results['reddit']['sentiment_f1_macro']:.4f}")
    print(f"    Emotion - Accuracy: {final_multitask_results['reddit']['emotion_accuracy']:.4f}, F1: {final_multitask_results['reddit']['emotion_f1_macro']:.4f}")
    print(f"    Combined - Accuracy: {final_multitask_results['reddit']['combined_accuracy']:.4f}, F1: {final_multitask_results['reddit']['combined_f1_macro']:.4f}")

# Compare with tuning results
print(f"\n📊 Comparison:")
print(f"  Tuning Combined F1 (on subset): {multitask_study.best_value:.4f}")
print(f"  Final Combined F1 (on full test): {final_multitask_results['combined_f1_macro']:.4f}")

# Clean up memory
aggressive_memory_cleanup()
print(f"💾 Final multitask model saved to: ./final_bertweet_multitask_model/")


# In[ ]:


print("\n" + "="*80)
print("📍 PHASE 4: COMPREHENSIVE RESULTS COMPARISON")
print("="*80)

print("\n📊 BERTWEET HYPERPARAMETER TUNING & FINAL TRAINING RESULTS")
print("="*60)

# Display hyperparameter tuning results
print(f"\n🎯 HYPERPARAMETER TUNING PERFORMANCE (on small subsets):")
print(f"  Sentiment Model:")
print(f"    Best F1 Score: {sentiment_study.best_value:.4f}")
print(f"    Key Parameters: LR={sentiment_study.best_params['learning_rate']:.2e}, Batch={sentiment_study.best_params['batch_size']}")

print(f"\n  Emotion Model:")
print(f"    Best F1 Score: {emotion_study.best_value:.4f}")
print(f"    Key Parameters: LR={emotion_study.best_params['learning_rate']:.2e}, Batch={emotion_study.best_params['batch_size']}")

print(f"\n  Multitask Model:")
print(f"    Best Combined F1 Score: {multitask_study.best_value:.4f}")
print(f"    Key Parameters: LR={multitask_study.best_params['learning_rate']:.2e}, Alpha={multitask_study.best_params['alpha']:.3f}")

# Display final model results (only using available metrics)
print(f"\n🏆 FINAL MODEL PERFORMANCE (on full test sets):")
print(f"  Sentiment Model:")
print(f"    General Dataset - Accuracy: {final_sentiment_results['general']['accuracy']:.4f}, F1: {final_sentiment_results['general']['f1_macro']:.4f}")
if final_sentiment_results.get('reddit'):
    print(f"    Reddit Dataset - Accuracy: {final_sentiment_results['reddit']['accuracy']:.4f}, F1: {final_sentiment_results['reddit']['f1_macro']:.4f}")

print(f"\n  Emotion Model:")
print(f"    General Dataset - Accuracy: {final_emotion_results['general']['accuracy']:.4f}, F1: {final_emotion_results['general']['f1_macro']:.4f}")
if final_emotion_results.get('reddit'):
    print(f"    Reddit Dataset - Accuracy: {final_emotion_results['reddit']['accuracy']:.4f}, F1: {final_emotion_results['reddit']['f1_macro']:.4f}")

print(f"\n  Multitask Model:")
print(f"    General Dataset:")
print(f"      Sentiment - Accuracy: {final_multitask_results['general']['sentiment_accuracy']:.4f}, F1: {final_multitask_results['general']['sentiment_f1_macro']:.4f}")
print(f"      Emotion - Accuracy: {final_multitask_results['general']['emotion_accuracy']:.4f}, F1: {final_multitask_results['general']['emotion_f1_macro']:.4f}")
print(f"      Combined - Accuracy: {final_multitask_results['general']['combined_accuracy']:.4f}, F1: {final_multitask_results['general']['combined_f1_macro']:.4f}")
if final_multitask_results.get('reddit'):
    print(f"    Reddit Dataset:")
    print(f"      Sentiment - Accuracy: {final_multitask_results['reddit']['sentiment_accuracy']:.4f}, F1: {final_multitask_results['reddit']['sentiment_f1_macro']:.4f}")
    print(f"      Emotion - Accuracy: {final_multitask_results['reddit']['emotion_accuracy']:.4f}, F1: {final_multitask_results['reddit']['emotion_f1_macro']:.4f}")
    print(f"      Combined - Accuracy: {final_multitask_results['reddit']['combined_accuracy']:.4f}, F1: {final_multitask_results['reddit']['combined_f1_macro']:.4f}")

# Performance comparison between tuning and final results
print(f"\n📈 TUNING vs FINAL PERFORMANCE COMPARISON:")
sentiment_improvement = final_sentiment_results['general']['f1_macro'] - sentiment_study.best_value
emotion_improvement = final_emotion_results['general']['f1_macro'] - emotion_study.best_value
multitask_improvement = final_multitask_results['general']['combined_f1_macro'] - multitask_study.best_value

print(f"  Sentiment:")
print(f"    Tuning F1 (subset): {sentiment_study.best_value:.4f}")
print(f"    Final F1 (full):    {final_sentiment_results['general']['f1_macro']:.4f}")
print(f"    Improvement:        {sentiment_improvement:+.4f} {'✅' if sentiment_improvement > 0 else '⚠️'}")

print(f"\n  Emotion:")
print(f"    Tuning F1 (subset): {emotion_study.best_value:.4f}")
print(f"    Final F1 (full):    {final_emotion_results['general']['f1_macro']:.4f}")
print(f"    Improvement:        {emotion_improvement:+.4f} {'✅' if emotion_improvement > 0 else '⚠️'}")

print(f"\n  Multitask:")
print(f"    Tuning Combined F1 (subset): {multitask_study.best_value:.4f}")
print(f"    Final Combined F1 (full):    {final_multitask_results['general']['combined_f1_macro']:.4f}")
print(f"    Improvement:                 {multitask_improvement:+.4f} {'✅' if multitask_improvement > 0 else '⚠️'}")

# Create comprehensive results summary
results_summary = {
    'model_type': 'BERTweet',
    'model_name': model_name,
    'pipeline_type': 'Fast Hyperparameter Tuning + Final Training',
    'hyperparameter_tuning': {
        'method': 'Fast Random Search',
        'subset_ratio': 0.02,
        'trials_per_model': 6,
        'epochs_per_trial': 2,
        'sentiment': {
            'best_f1': float(sentiment_study.best_value),
            'best_params': sentiment_study.best_params
        },
        'emotion': {
            'best_f1': float(emotion_study.best_value),
            'best_params': emotion_study.best_params
        },
        'multitask': {
            'best_combined_f1': float(multitask_study.best_value),
            'best_params': multitask_study.best_params
        }
    },
    'final_models': {
        'sentiment': final_sentiment_results,
        'emotion': final_emotion_results,
        'multitask': final_multitask_results
    },
    'performance_improvements': {
        'sentiment_f1_improvement': float(sentiment_improvement),
        'emotion_f1_improvement': float(emotion_improvement),
        'multitask_f1_improvement': float(multitask_improvement)
    },
    'model_locations': {
        'sentiment': './final_bertweet_sentiment_model/',
        'emotion': './final_bertweet_emotion_model/',
        'multitask': './final_bertweet_multitask_model/'
    }
}

# Save results
import json
with open('comprehensive_bertweet_results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n📁 MODEL LOCATIONS:")
print(f"  📦 Sentiment model: ./final_bertweet_sentiment_model/")
print(f"  📦 Emotion model: ./final_bertweet_emotion_model/")
print(f"  📦 Multitask model: ./final_bertweet_multitask_model/")

print(f"\n📄 RESULTS SAVED:")
print(f"  📊 Comprehensive summary: ./comprehensive_bertweet_results_summary.json")

print(f"\n🎉 COMPLETE BERTWEET PIPELINE FINISHED!")
print(f"✅ Fast hyperparameter tuning + optimized final training completed!")
print(f"🚀 Pipeline completed in a fraction of the original time!")

# Add comprehensive results comparison
print(f"\n" + "="*80)
print(f"🏁 COMPREHENSIVE BERTWEET RESULTS COMPARISON")
print(f"="*80)

print(f"\n📊 BERTWEET MODEL PERFORMANCE COMPARISON:")
print(f"  {'='*60}")

# Sentiment Model Comparison
print(f"\n🎯 SENTIMENT MODEL:")
print(f"  Initial Model:")
print(f"    General Dataset - Accuracy: {all_results['initial_sentiment']['general']['accuracy']:.4f}, F1: {all_results['initial_sentiment']['general']['f1_macro']:.4f}")
if all_results['initial_sentiment'].get('reddit'):
    print(f"    Reddit Dataset - Accuracy: {all_results['initial_sentiment']['reddit']['accuracy']:.4f}, F1: {all_results['initial_sentiment']['reddit']['f1_macro']:.4f}")

print(f"  Final Optimized Model:")
print(f"    General Dataset - Accuracy: {final_sentiment_results['general']['accuracy']:.4f}, F1: {final_sentiment_results['general']['f1_macro']:.4f}")
if final_sentiment_results.get('reddit'):
    print(f"    Reddit Dataset - Accuracy: {final_sentiment_results['reddit']['accuracy']:.4f}, F1: {final_sentiment_results['reddit']['f1_macro']:.4f}")

# Calculate improvements
sentiment_general_improvement = final_sentiment_results['general']['accuracy'] - all_results['initial_sentiment']['general']['accuracy']
sentiment_f1_improvement = final_sentiment_results['general']['f1_macro'] - all_results['initial_sentiment']['general']['f1_macro']
print(f"  Improvements:")
print(f"    General Accuracy: {sentiment_general_improvement:+.4f}")
print(f"    General F1: {sentiment_f1_improvement:+.4f}")

# Emotion Model Comparison
print(f"\n🎭 EMOTION MODEL:")
print(f"  Initial Model:")
print(f"    General Dataset - Accuracy: {all_results['initial_emotion']['general']['accuracy']:.4f}, F1: {all_results['initial_emotion']['general']['f1_macro']:.4f}")
if all_results['initial_emotion'].get('reddit'):
    print(f"    Reddit Dataset - Accuracy: {all_results['initial_emotion']['reddit']['accuracy']:.4f}, F1: {all_results['initial_emotion']['reddit']['f1_macro']:.4f}")

print(f"  Final Optimized Model:")
print(f"    General Dataset - Accuracy: {final_emotion_results['general']['accuracy']:.4f}, F1: {final_emotion_results['general']['f1_macro']:.4f}")
if final_emotion_results.get('reddit'):
    print(f"    Reddit Dataset - Accuracy: {final_emotion_results['reddit']['accuracy']:.4f}, F1: {final_emotion_results['reddit']['f1_macro']:.4f}")

# Calculate improvements
emotion_general_improvement = final_emotion_results['general']['accuracy'] - all_results['initial_emotion']['general']['accuracy']
emotion_f1_improvement = final_emotion_results['general']['f1_macro'] - all_results['initial_emotion']['general']['f1_macro']
print(f"  Improvements:")
print(f"    General Accuracy: {emotion_general_improvement:+.4f}")
print(f"    General F1: {emotion_f1_improvement:+.4f}")

# Multitask Model Comparison
print(f"\n🔄 MULTITASK MODEL:")
print(f"  Initial Model:")
print(f"    General Dataset:")
print(f"      Sentiment - Accuracy: {all_results['initial_multitask']['general']['sentiment_accuracy']:.4f}, F1: {all_results['initial_multitask']['general']['sentiment_f1_macro']:.4f}")
print(f"      Emotion - Accuracy: {all_results['initial_multitask']['general']['emotion_accuracy']:.4f}, F1: {all_results['initial_multitask']['general']['emotion_f1_macro']:.4f}")
print(f"      Combined - Accuracy: {all_results['initial_multitask']['general']['combined_accuracy']:.4f}, F1: {all_results['initial_multitask']['general']['combined_f1_macro']:.4f}")
if all_results['initial_multitask'].get('reddit'):
    print(f"    Reddit Dataset:")
    print(f"      Sentiment - Accuracy: {all_results['initial_multitask']['reddit']['sentiment_accuracy']:.4f}, F1: {all_results['initial_multitask']['reddit']['sentiment_f1_macro']:.4f}")
    print(f"      Emotion - Accuracy: {all_results['initial_multitask']['reddit']['emotion_accuracy']:.4f}, F1: {all_results['initial_multitask']['reddit']['emotion_f1_macro']:.4f}")
    print(f"      Combined - Accuracy: {all_results['initial_multitask']['reddit']['combined_accuracy']:.4f}, F1: {all_results['initial_multitask']['reddit']['combined_f1_macro']:.4f}")

print(f"  Final Optimized Model:")
print(f"    General Dataset:")
print(f"      Sentiment - Accuracy: {final_multitask_results['general']['sentiment_accuracy']:.4f}, F1: {final_multitask_results['general']['sentiment_f1_macro']:.4f}")
print(f"      Emotion - Accuracy: {final_multitask_results['general']['emotion_accuracy']:.4f}, F1: {final_multitask_results['general']['emotion_f1_macro']:.4f}")
print(f"      Combined - Accuracy: {final_multitask_results['general']['combined_accuracy']:.4f}, F1: {final_multitask_results['general']['combined_f1_macro']:.4f}")
if final_multitask_results.get('reddit'):
    print(f"    Reddit Dataset:")
    print(f"      Sentiment - Accuracy: {final_multitask_results['reddit']['sentiment_accuracy']:.4f}, F1: {final_multitask_results['reddit']['sentiment_f1_macro']:.4f}")
    print(f"      Emotion - Accuracy: {final_multitask_results['reddit']['emotion_accuracy']:.4f}, F1: {final_multitask_results['reddit']['emotion_f1_macro']:.4f}")
    print(f"      Combined - Accuracy: {final_multitask_results['reddit']['combined_accuracy']:.4f}, F1: {final_multitask_results['reddit']['combined_f1_macro']:.4f}")

# Calculate improvements
multitask_sentiment_improvement = final_multitask_results['general']['sentiment_accuracy'] - all_results['initial_multitask']['general']['sentiment_accuracy']
multitask_emotion_improvement = final_multitask_results['general']['emotion_accuracy'] - all_results['initial_multitask']['general']['emotion_accuracy']
multitask_combined_improvement = final_multitask_results['general']['combined_accuracy'] - all_results['initial_multitask']['general']['combined_accuracy']
print(f"  Improvements:")
print(f"    Sentiment Accuracy: {multitask_sentiment_improvement:+.4f}")
print(f"    Emotion Accuracy: {multitask_emotion_improvement:+.4f}")
print(f"    Combined Accuracy: {multitask_combined_improvement:+.4f}")

print(f"\n🎉 BERTWEET TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print(f"   All models trained and evaluated on both general and Reddit datasets")
print(f"   Hyperparameter optimization completed using macro F1 on general datasets")
print(f"   Final models saved and ready for deployment")

# Display final summary table
print(f"\n" + "="*80)
print(f"📋 FINAL PERFORMANCE SUMMARY")
print(f"="*80)
print(f"{'Model':<12} {'Accuracy':<10} {'F1 Score':<10} {'Improvement':<12} {'Status':<10}")
print(f"-" * 65)
print(f"{'Sentiment':<12} {final_sentiment_results['general']['accuracy']:<10.4f} {final_sentiment_results['general']['f1_macro']:<10.4f} {sentiment_improvement:+10.4f} {'✅ Complete':<10}")
print(f"{'Emotion':<12} {final_emotion_results['general']['accuracy']:<10.4f} {final_emotion_results['general']['f1_macro']:<10.4f} {emotion_improvement:+10.4f} {'✅ Complete':<10}")
print(f"{'Multitask':<12} {final_multitask_results['general']['combined_accuracy']:<10.4f} {final_multitask_results['general']['combined_f1_macro']:<10.4f} {multitask_improvement:+10.4f} {'✅ Complete':<10}")
print("="*80)

print(f"\n🐦 BERTweet models are ready for social media text processing!")
print(f"💡 All models trained with optimized hyperparameters found via fast search!")


# %%
