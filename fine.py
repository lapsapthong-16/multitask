import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

class BERTweetFineTuner:
    def __init__(self):
        self.model_name = "vinai/bertweet-base"
        self.tokenizer = None
        self.sentiment_model = None
        self.emotion_model = None
        
        # GoEmotions emotion labels (27 classes)
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
    def setup_tokenizer(self):
        """Initialize the BERTweet tokenizer"""
        print("🔧 Setting up BERTweet tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer ready!")
        
    def load_sentiment_dataset(self):
        """Load and prepare sentiment dataset (SST-2)"""
        print("📚 Loading SST-2 sentiment dataset...")
        
        try:
            # Load SST-2 dataset from Hugging Face
            dataset = load_dataset("sst2")
            
            def preprocess_sentiment(examples):
                # Tokenize texts
                tokenized = self.tokenizer(
                    examples["sentence"], 
                    truncation=True, 
                    padding=True,
                    max_length=128
                )
                
                # Keep original labels (0=Negative, 1=Positive)
                tokenized["labels"] = examples["label"]
                
                return tokenized
            
            # Process the dataset
            tokenized_dataset = dataset.map(preprocess_sentiment, batched=True)
            
            # Use smaller subsets for faster training
            train_subset = tokenized_dataset["train"].shuffle(seed=42).select(range(10000))
            val_subset = tokenized_dataset["validation"].shuffle(seed=42).select(range(2000))
            
            print(f"✅ SST-2 dataset loaded:")
            print(f"   Training samples: {len(train_subset)}")
            print(f"   Validation samples: {len(val_subset)}")
            
            return {"train": train_subset, "validation": val_subset}
            
        except Exception as e:
            print(f"❌ Error loading SST-2: {e}")
            return self.create_dummy_sentiment_dataset()
    
    def load_emotion_dataset(self):
        """Load and prepare emotion dataset (GoEmotions)"""
        print("💭 Loading GoEmotions dataset...")
        
        try:
            # Load GoEmotions dataset
            dataset = load_dataset("go_emotions", "simplified")
            
            def preprocess_emotion(examples):
                # Tokenize texts
                tokenized = self.tokenizer(
                    examples["text"], 
                    truncation=True, 
                    padding=True,
                    max_length=128
                )
                
                # Use the first emotion label (multi-label -> single label)
                labels = []
                for label_list in examples["labels"]:
                    if label_list:  # If there are labels
                        primary_label = label_list[0]  # Take first label
                        labels.append(primary_label)
                    else:  # No labels -> neutral
                        labels.append(27)  # neutral
                
                tokenized["labels"] = labels
                
                return tokenized
            
            # Process the dataset
            tokenized_dataset = dataset.map(preprocess_emotion, batched=True)
            
            # Filter to get a smaller subset for faster training
            train_subset = tokenized_dataset["train"].shuffle(seed=42).select(range(10000))
            val_subset = tokenized_dataset["validation"].shuffle(seed=42).select(range(2000))
            
            print(f"✅ GoEmotions dataset loaded:")
            print(f"   Training samples: {len(train_subset)}")
            print(f"   Validation samples: {len(val_subset)}")
            print(f"   Emotion classes: {len(self.emotion_labels)}")
            
            return {"train": train_subset, "validation": val_subset}
            
        except Exception as e:
            print(f"❌ Error loading GoEmotions: {e}")
            return self.create_dummy_emotion_dataset()
    
    def create_dummy_sentiment_dataset(self):
        """Create a small dummy sentiment dataset for testing"""
        print("🔄 Creating dummy sentiment dataset...")
        dummy_data = {
            "sentence": [
                "I love this product!", "This is terrible", "It's okay", 
                "Amazing quality", "Worst experience ever", "Not bad"
            ] * 200,
            "label": [1, 0, 1, 1, 0, 1] * 200
        }
        
        def preprocess_dummy(examples):
            tokenized = self.tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)
            tokenized["labels"] = examples["label"]
            return tokenized
        
        dataset = Dataset.from_dict(dummy_data)
        tokenized = dataset.map(preprocess_dummy, batched=True)
        
        return {"train": tokenized, "validation": tokenized.select(range(100))}
    
    def create_dummy_emotion_dataset(self):
        """Create a small dummy emotion dataset for testing"""
        print("🔄 Creating dummy emotion dataset...")
        emotions_text = [
            "I'm so happy!", "This is sad", "I'm angry", "That's scary", 
            "What a surprise!", "Okay", "I admire you", "That's funny",
            "I'm confused", "I'm curious", "I want this", "I'm disappointed"
        ]
        emotions_labels = [17, 25, 2, 14, 26, 27, 0, 1, 6, 7, 8, 9]  # joy, sadness, anger, fear, surprise, neutral, etc.
        
        dummy_data = {
            "text": emotions_text * 100,
            "labels": emotions_labels * 100
        }
        
        def preprocess_dummy(examples):
            tokenized = self.tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
            tokenized["labels"] = examples["labels"]
            return tokenized
        
        dataset = Dataset.from_dict(dummy_data)
        tokenized = dataset.map(preprocess_dummy, batched=True)
        
        return {"train": tokenized, "validation": tokenized.select(range(120))}
    
    def fine_tune_sentiment_model(self, dataset):
        """Fine-tune BERTweet for sentiment classification"""
        print("🎯 Fine-tuning BERTweet for sentiment classification...")
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2,  # Positive, Negative
            ignore_mismatched_sizes=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./sentiment_model",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            report_to=None  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
            accuracy = accuracy_score(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("🚀 Starting sentiment model training...")
        trainer.train()
        
        # Save model
        os.makedirs("./sentiment_model_final", exist_ok=True)
        trainer.save_model("./sentiment_model_final")
        self.sentiment_model = model
        
        print("✅ Sentiment model training completed!")
        return trainer
    
    def fine_tune_emotion_model(self, dataset):
        """Fine-tune BERTweet for emotion classification"""
        print("💭 Fine-tuning BERTweet for emotion classification...")
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.emotion_labels),
            ignore_mismatched_sizes=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./emotion_model",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            report_to=None  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
            accuracy = accuracy_score(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("🚀 Starting emotion model training...")
        trainer.train()
        
        # Save model
        os.makedirs("./emotion_model_final", exist_ok=True)
        trainer.save_model("./emotion_model_final")
        self.emotion_model = model
        
        print("✅ Emotion model training completed!")
        return trainer
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            
            # Map to label (0=Negative, 1=Positive)
            sentiment_map = {0: "Negative", 1: "Positive"}
            predicted_label = sentiment_map.get(predicted_class, "Unknown")
            
            return predicted_label, confidence
    
    def predict_emotion(self, text):
        """Predict emotion for a single text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            
            # Map to emotion label
            if predicted_class < len(self.emotion_labels):
                predicted_label = self.emotion_labels[predicted_class].title()
            else:
                predicted_label = "Neutral"
            
            return predicted_label, confidence
    
    def annotate_reddit_data(self, csv_path):
        """Annotate Reddit data with fine-tuned models"""
        print("📝 Annotating Reddit data with fine-tuned models...")
        
        # Load cleaned Reddit data
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded Reddit data: {len(df)} samples")
        except Exception as e:
            print(f"❌ Error loading Reddit data: {e}")
            return None
        
        # Make predictions
        print("🔮 Making predictions...")
        sentiment_predictions = []
        emotion_predictions = []
        sentiment_confidences = []
        emotion_confidences = []
        
        for idx, text in enumerate(df['text_content']):
            if idx % 10 == 0:
                print(f"   Processing {idx+1}/{len(df)} samples...")
            
            # Predict sentiment
            sent_pred, sent_conf = self.predict_sentiment(text)
            sentiment_predictions.append(sent_pred)
            sentiment_confidences.append(sent_conf)
            
            # Predict emotion
            emo_pred, emo_conf = self.predict_emotion(text)
            emotion_predictions.append(emo_pred)
            emotion_confidences.append(emo_conf)
        
        # Add predictions to dataframe
        df['predicted_sentiment'] = sentiment_predictions
        df['predicted_emotion'] = emotion_predictions
        df['sentiment_confidence'] = sentiment_confidences
        df['emotion_confidence'] = emotion_confidences
        
        # Save annotated data
        df.to_csv('annotated_reddit_predictions.csv', index=False)
        print("💾 Annotated predictions saved to: annotated_reddit_predictions.csv")
        
        return df
    
    def evaluate_on_gold_standard(self, annotated_csv_path):
        """Evaluate fine-tuned models against gold standard labels"""
        print("📊 Evaluating against gold standard labels...")
        
        # Load annotated Reddit data with gold standard
        try:
            df = pd.read_csv(annotated_csv_path)
            print(f"✅ Loaded annotated Reddit data: {len(df)} samples")
        except Exception as e:
            print(f"❌ Error loading annotated Reddit data: {e}")
            return None, None
        
        # Check required columns
        required_cols = ['text_content', 'sentiment_bertweet', 'emotion_bertweet']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None, None
        
        # Make predictions on the text content
        print("🔮 Making predictions on gold standard data...")
        sentiment_predictions = []
        emotion_predictions = []
        
        for idx, text in enumerate(df['text_content']):
            if idx % 10 == 0:
                print(f"   Processing {idx+1}/{len(df)} samples...")
            
            # Predict sentiment
            sent_pred, _ = self.predict_sentiment(text)
            sentiment_predictions.append(sent_pred)
            
            # Predict emotion
            emo_pred, _ = self.predict_emotion(text)
            emotion_predictions.append(emo_pred)
        
        # Add predictions to dataframe
        df['predicted_sentiment'] = sentiment_predictions
        df['predicted_emotion'] = emotion_predictions
        
        # Evaluate sentiment
        sentiment_results = self.evaluate_sentiment_predictions(df)
        
        # Evaluate emotion
        emotion_results = self.evaluate_emotion_predictions(df)
        
        # Create visualizations
        self.create_evaluation_plots(sentiment_results, emotion_results, df)
        
        # Save results
        df.to_csv('final_evaluation_results.csv', index=False)
        print("💾 Final evaluation results saved to: final_evaluation_results.csv")
        
        return sentiment_results, emotion_results
    
    def evaluate_sentiment_predictions(self, df):
        """Evaluate sentiment predictions against gold standard"""
        print("\n🎯 SENTIMENT CLASSIFICATION RESULTS:")
        print("=" * 50)
        
        true_labels = df['sentiment_bertweet'].tolist()
        predicted_labels = df['predicted_sentiment'].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted', zero_division=0
        )
        
        # Get unique labels for confusion matrix
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'unique_labels': unique_labels,
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
        }
        
        # Print metrics
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Precision (macro):  {precision_macro:.4f}")
        print(f"Recall (macro):     {recall_macro:.4f}")
        print(f"F1-score (macro):   {f1_macro:.4f}")
        print(f"F1-score (weighted): {f1_weighted:.4f}")
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predicted_labels, zero_division=0))
        
        return results
    
    def evaluate_emotion_predictions(self, df):
        """Evaluate emotion predictions against gold standard"""
        print("\n💭 EMOTION CLASSIFICATION RESULTS:")
        print("=" * 50)
        
        true_labels = df['emotion_bertweet'].tolist()
        predicted_labels = df['predicted_emotion'].tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted', zero_division=0
        )
        
        # Get unique labels for confusion matrix
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'unique_labels': unique_labels,
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
        }
        
        # Print metrics
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Precision (macro):  {precision_macro:.4f}")
        print(f"Recall (macro):     {recall_macro:.4f}")
        print(f"F1-score (macro):   {f1_macro:.4f}")
        print(f"F1-score (weighted): {f1_weighted:.4f}")
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predicted_labels, zero_division=0))
        
        return results
    
    def create_evaluation_plots(self, sentiment_results, emotion_results, df):
        """Create confusion matrices and evaluation plots"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Sentiment confusion matrix
        sentiment_cm = sentiment_results['confusion_matrix']
        sentiment_labels = sentiment_results['unique_labels']
        
        sns.heatmap(sentiment_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sentiment_labels, yticklabels=sentiment_labels, ax=axes[0])
        axes[0].set_title('Sentiment Classification\n(Fine-tuned BERTweet)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('True', fontsize=12)
        
        # Emotion confusion matrix
        emotion_cm = emotion_results['confusion_matrix']
        emotion_labels = emotion_results['unique_labels']
        
        # Limit emotion labels for readability if too many
        if len(emotion_labels) > 10:
            # Show only the most common emotions
            emotion_counts = pd.Series(emotion_results['true_labels']).value_counts()
            top_emotions = emotion_counts.head(10).index.tolist()
            
            # Filter confusion matrix and labels
            emotion_label_indices = [i for i, label in enumerate(emotion_labels) if label in top_emotions]
            filtered_cm = emotion_cm[np.ix_(emotion_label_indices, emotion_label_indices)]
            filtered_labels = [emotion_labels[i] for i in emotion_label_indices]
            
            sns.heatmap(filtered_cm, annot=True, fmt='d', cmap='Greens', 
                       xticklabels=filtered_labels, yticklabels=filtered_labels, ax=axes[1])
            axes[1].set_title('Emotion Classification (Top 10)\n(Fine-tuned BERTweet)', fontsize=14, fontweight='bold')
        else:
            sns.heatmap(emotion_cm, annot=True, fmt='d', cmap='Greens', 
                       xticklabels=emotion_labels, yticklabels=emotion_labels, ax=axes[1])
            axes[1].set_title('Emotion Classification\n(Fine-tuned BERTweet)', fontsize=14, fontweight='bold')
        
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('True', fontsize=12)
        
        # Rotate labels for better readability
        axes[0].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig('fine_tuned_bertweet_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📈 Evaluation plots saved to: fine_tuned_bertweet_evaluation.png")
    
    def create_summary_report(self, sentiment_results, emotion_results):
        """Create a summary report of the evaluation"""
        print("\n📋 EVALUATION SUMMARY REPORT:")
        print("=" * 60)
        
        # Create summary table
        summary_data = {
            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'F1-Score (Weighted)'],
            'Sentiment': [
                f"{sentiment_results['accuracy']:.4f}",
                f"{sentiment_results['precision_macro']:.4f}",
                f"{sentiment_results['recall_macro']:.4f}",
                f"{sentiment_results['f1_macro']:.4f}",
                f"{sentiment_results['f1_weighted']:.4f}"
            ],
            'Emotion': [
                f"{emotion_results['accuracy']:.4f}",
                f"{emotion_results['precision_macro']:.4f}",
                f"{emotion_results['recall_macro']:.4f}",
                f"{emotion_results['f1_macro']:.4f}",
                f"{emotion_results['f1_weighted']:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv('evaluation_summary.csv', index=False)
        print("\n💾 Summary saved to: evaluation_summary.csv")
        
        return summary_df

def main():
    """Main execution function for BERTweet fine-tuning and evaluation"""
    print("🚀 BERTweet FINE-TUNING AND EVALUATION PIPELINE")
    print("=" * 60)
    
    # Initialize fine-tuner
    fine_tuner = BERTweetFineTuner()
    
    # Setup tokenizer
    fine_tuner.setup_tokenizer()
    
    try:
        # Step 1: Load datasets for fine-tuning
        print("\n📚 STEP 1: Loading training datasets...")
        sentiment_dataset = fine_tuner.load_sentiment_dataset()
        emotion_dataset = fine_tuner.load_emotion_dataset()
        
        # Step 2: Fine-tune sentiment model
        print("\n🎯 STEP 2: Fine-tuning sentiment model...")
        sentiment_trainer = fine_tuner.fine_tune_sentiment_model(sentiment_dataset)
        
        # Step 3: Fine-tune emotion model
        print("\n💭 STEP 3: Fine-tuning emotion model...")
        emotion_trainer = fine_tuner.fine_tune_emotion_model(emotion_dataset)
        
        # Step 4: Annotate cleaned Reddit data
        print("\n📝 STEP 4: Annotating cleaned Reddit data...")
        annotated_df = fine_tuner.annotate_reddit_data('cleaned_reddit_posts.csv')
        
        # Step 5: Evaluate against gold standard
        print("\n📊 STEP 5: Evaluating against gold standard...")
        sentiment_results, emotion_results = fine_tuner.evaluate_on_gold_standard('annotated_reddit_posts.csv')
        
        if sentiment_results is None or emotion_results is None:
            print("❌ Evaluation failed. Exiting...")
            return None
        
        # Step 6: Create summary report
        print("\n📋 STEP 6: Creating summary report...")
        summary_df = fine_tuner.create_summary_report(sentiment_results, emotion_results)
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("📋 Summary: BERTweet models fine-tuned and evaluated on Reddit PR-crisis data")
        print("📁 Output files:")
        print("   - ./sentiment_model_final/ (fine-tuned sentiment model)")
        print("   - ./emotion_model_final/ (fine-tuned emotion model)")
        print("   - annotated_reddit_predictions.csv (annotated cleaned data)")
        print("   - final_evaluation_results.csv (evaluation results)")
        print("   - evaluation_summary.csv (metrics summary)")
        print("   - fine_tuned_bertweet_evaluation.png (confusion matrices)")
        
        return summary_df
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()