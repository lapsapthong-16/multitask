import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_models():
    """Initialize all models and tokenizers"""
    print("🔧 Setting up models...")
    
    # VADER analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # BERTweet models
    print("📡 Loading BERTweet models...")
    bertweet_sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    bertweet_emotion_model = "j-hartmann/emotion-english-distilroberta-base"
    
    # RoBERTa models  
    print("🤖 Loading RoBERTa models...")
    roberta_sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    roberta_emotion_model = "j-hartmann/emotion-english-distilroberta-base"
    
    # Create pipelines
    bertweet_sentiment_pipe = pipeline("sentiment-analysis", 
                                      model=bertweet_sentiment_model,
                                      tokenizer=bertweet_sentiment_model,
                                      max_length=512, 
                                      truncation=True)
    
    bertweet_emotion_pipe = pipeline("text-classification", 
                                   model=bertweet_emotion_model,
                                   tokenizer=bertweet_emotion_model,
                                   max_length=512,
                                   truncation=True)
    
    roberta_sentiment_pipe = pipeline("sentiment-analysis", 
                                    model=roberta_sentiment_model,
                                    tokenizer=roberta_sentiment_model,
                                    max_length=512,
                                    truncation=True)
    
    roberta_emotion_pipe = pipeline("text-classification", 
                                  model=roberta_emotion_model,
                                  tokenizer=roberta_emotion_model,
                                  max_length=512,
                                  truncation=True)
    
    print("✅ All models loaded successfully!")
    return vader_analyzer, bertweet_sentiment_pipe, bertweet_emotion_pipe, roberta_sentiment_pipe, roberta_emotion_pipe

def analyze_vader_sentiment(text, analyzer):
    """Analyze sentiment using VADER"""
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_bertweet_sentiment(text, pipe):
    """Analyze sentiment using BERTweet"""
    try:
        result = pipe(text)[0]
        label = result['label']
        # Map labels to consistent format
        if label in ['LABEL_0', 'NEGATIVE']:
            return "Negative"
        elif label in ['LABEL_1', 'NEUTRAL']:
            return "Neutral"
        elif label in ['LABEL_2', 'POSITIVE']:
            return "Positive"
        else:
            return label.title()
    except Exception as e:
        print(f"Error in BERTweet sentiment: {e}")
        return "Neutral"

def analyze_emotion(text, pipe):
    """Analyze emotion using emotion classification model"""
    try:
        result = pipe(text)[0]
        emotion = result['label']
        # Map to consistent emotion labels
        emotion_mapping = {
            'joy': 'Joy',
            'sadness': 'Sadness', 
            'anger': 'Anger',
            'fear': 'Fear',
            'surprise': 'Surprise',
            'disgust': 'Disgust',
            'love': 'Love',
            'neutral': 'Neutral'
        }
        return emotion_mapping.get(emotion.lower(), emotion.title())
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return "Neutral"

def analyze_roberta_sentiment(text, pipe):
    """Analyze sentiment using RoBERTa"""
    try:
        result = pipe(text)[0]
        label = result['label']
        # Map labels to consistent format
        if label in ['LABEL_0', 'NEGATIVE']:
            return "Negative"
        elif label in ['LABEL_1', 'NEUTRAL']:
            return "Neutral"
        elif label in ['LABEL_2', 'POSITIVE']:
            return "Positive"
        else:
            return label.title()
    except Exception as e:
        print(f"Error in RoBERTa sentiment: {e}")
        return "Neutral"

def process_dataset(input_file, output_file):
    """Process the entire dataset and add annotations"""
    print(f"📂 Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"📊 Dataset loaded: {len(df)} entries")
    
    # Setup models
    vader_analyzer, bertweet_sentiment_pipe, bertweet_emotion_pipe, roberta_sentiment_pipe, roberta_emotion_pipe = setup_models()
    
    # Initialize new columns
    df['sentiment_vader'] = ""
    df['emotion_vader'] = "NA"  # VADER doesn't do emotion
    df['sentiment_bertweet'] = ""
    df['emotion_bertweet'] = ""
    df['sentiment_roberta'] = ""
    df['emotion_roberta'] = ""
    
    print("🔄 Processing annotations...")
    
    # Process each row with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
        text = str(row['text_content'])
        
        # Skip if text is empty or too short
        if len(text.strip()) < 3:
            df.at[idx, 'sentiment_vader'] = "Neutral"
            df.at[idx, 'sentiment_bertweet'] = "Neutral"
            df.at[idx, 'emotion_bertweet'] = "Neutral"
            df.at[idx, 'sentiment_roberta'] = "Neutral"
            df.at[idx, 'emotion_roberta'] = "Neutral"
            continue
        
        # 1. VADER Analysis
        df.at[idx, 'sentiment_vader'] = analyze_vader_sentiment(text, vader_analyzer)
        
        # 2. BERTweet Analysis
        df.at[idx, 'sentiment_bertweet'] = analyze_bertweet_sentiment(text, bertweet_sentiment_pipe)
        df.at[idx, 'emotion_bertweet'] = analyze_emotion(text, bertweet_emotion_pipe)
        
        # 3. RoBERTa Analysis
        df.at[idx, 'sentiment_roberta'] = analyze_roberta_sentiment(text, roberta_sentiment_pipe)
        df.at[idx, 'emotion_roberta'] = analyze_emotion(text, roberta_emotion_pipe)
    
    # Save annotated dataset
    print(f"💾 Saving annotated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"✅ Complete! Annotated dataset saved with {len(df)} entries")
    
    # Print summary statistics
    print("\n📈 ANNOTATION SUMMARY:")
    print("=" * 50)
    
    print("\n🎯 VADER Sentiment Distribution:")
    print(df['sentiment_vader'].value_counts())
    
    print("\n🐦 BERTweet Sentiment Distribution:")
    print(df['sentiment_bertweet'].value_counts())
    
    print("\n🐦 BERTweet Emotion Distribution:")
    print(df['emotion_bertweet'].value_counts())
    
    print("\n🤖 RoBERTa Sentiment Distribution:")
    print(df['sentiment_roberta'].value_counts())
    
    print("\n🤖 RoBERTa Emotion Distribution:")
    print(df['emotion_roberta'].value_counts())
    
    return df

def main():
    """Main execution function"""
    print("🚀 Starting Sentiment & Emotion Annotation Pipeline")
    print("=" * 55)
    
    input_file = "cleaned_reddit_posts.csv"
    output_file = "annotated_reddit_posts.csv"
    
    try:
        annotated_df = process_dataset(input_file, output_file)
        print(f"\n🎉 SUCCESS! Annotated dataset ready: {output_file}")
        
        # Show sample of annotated data
        print("\n📋 Sample of annotated data:")
        sample_cols = ['id', 'text_content', 'sentiment_vader', 'sentiment_bertweet', 
                      'emotion_bertweet', 'sentiment_roberta', 'emotion_roberta']
        print(annotated_df[sample_cols].head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main() 