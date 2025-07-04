import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def interpret_kappa(kappa_value):
    if kappa_value > 0.9:
        return "Perfect agreement"
    elif 0.8 <= kappa_value <= 0.9:
        return "Strong agreement"  
    elif 0.6 <= kappa_value < 0.8:
        return "Moderate agreement"
    elif 0.4 <= kappa_value < 0.6:
        return "Weak agreement"
    elif 0.2 <= kappa_value < 0.4:
        return "Fair agreement"
    else:
        return "Poor agreement"

def calculate_kappa_for_sentiments(df):
    """Calculate Cohen's Kappa for sentiment annotations"""
    print("🎯 TASK 1: COHEN'S KAPPA ANALYSIS")
    print("=" * 50)
    
    # Sentiment comparisons
    sentiment_pairs = [
        ('sentiment_vader', 'sentiment_bertweet'),
        ('sentiment_vader', 'sentiment_roberta'),
        ('sentiment_bertweet', 'sentiment_roberta')
    ]
    
    kappa_results = []
    
    print("\n📊 SENTIMENT AGREEMENT ANALYSIS:")
    print("-" * 40)
    
    for col1, col2 in sentiment_pairs:
        # Filter out any missing or invalid values
        valid_mask = (df[col1].notna()) & (df[col2].notna()) & \
                    (df[col1] != '') & (df[col2] != '')
        
        if valid_mask.sum() == 0:
            print(f"❌ No valid data for {col1} vs {col2}")
            continue
            
        y1 = df.loc[valid_mask, col1]
        y2 = df.loc[valid_mask, col2]
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(y1, y2)
        interpretation = interpret_kappa(kappa)
        
        kappa_results.append({
            'Comparison': f"{col1.replace('sentiment_', '').upper()} vs {col2.replace('sentiment_', '').upper()}",
            'Kappa': kappa,
            'Interpretation': interpretation,
            'Valid_Samples': len(y1)
        })
        
        print(f"\n🔸 {col1.replace('sentiment_', '').upper()} vs {col2.replace('sentiment_', '').upper()}:")
        print(f"   Cohen's Kappa: {kappa:.4f}")
        print(f"   Interpretation: {interpretation}")
        print(f"   Valid samples: {len(y1)}")
        
        # Show distribution
        print(f"   {col1.replace('sentiment_', '').upper()} distribution: {dict(Counter(y1))}")
        print(f"   {col2.replace('sentiment_', '').upper()} distribution: {dict(Counter(y2))}")
    
    return kappa_results

def calculate_kappa_for_emotions(df):
    """Calculate Cohen's Kappa for emotion annotations"""
    print("\n\n💭 EMOTION AGREEMENT ANALYSIS:")
    print("-" * 40)
    
    # Emotion comparison (only BERTweet vs RoBERTa since VADER doesn't do emotions)
    emotion_pairs = [('emotion_bertweet', 'emotion_roberta')]
    
    emotion_kappa_results = []
    
    for col1, col2 in emotion_pairs:
        # Filter out NA and missing values
        valid_mask = (df[col1].notna()) & (df[col2].notna()) & \
                    (df[col1] != 'NA') & (df[col2] != 'NA') & \
                    (df[col1] != '') & (df[col2] != '')
        
        if valid_mask.sum() == 0:
            print(f"❌ No valid data for {col1} vs {col2}")
            continue
            
        y1 = df.loc[valid_mask, col1]
        y2 = df.loc[valid_mask, col2]
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(y1, y2)
        interpretation = interpret_kappa(kappa)
        
        emotion_kappa_results.append({
            'Comparison': f"{col1.replace('emotion_', '').upper()} vs {col2.replace('emotion_', '').upper()}",
            'Kappa': kappa,
            'Interpretation': interpretation,
            'Valid_Samples': len(y1)
        })
        
        print(f"\n🔸 {col1.replace('emotion_', '').upper()} vs {col2.replace('emotion_', '').upper()}:")
        print(f"   Cohen's Kappa: {kappa:.4f}")
        print(f"   Interpretation: {interpretation}")
        print(f"   Valid samples: {len(y1)}")
        
        # Show emotion distributions
        print(f"   {col1.replace('emotion_', '').upper()} distribution: {dict(Counter(y1))}")
        print(f"   {col2.replace('emotion_', '').upper()} distribution: {dict(Counter(y2))}")
    
    return emotion_kappa_results

def create_confusion_matrices(df):
    """Create confusion matrices for visual analysis"""
    from sklearn.metrics import confusion_matrix
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for Inter-Annotator Agreement', fontsize=16, fontweight='bold')
    
    # Sentiment comparisons
    sentiment_pairs = [
        ('sentiment_vader', 'sentiment_bertweet', 'VADER vs BERTweet'),
        ('sentiment_vader', 'sentiment_roberta', 'VADER vs RoBERTa'),
        ('sentiment_bertweet', 'sentiment_roberta', 'BERTweet vs RoBERTa')
    ]
    
    for i, (col1, col2, title) in enumerate(sentiment_pairs):
        if i >= 3:  # Only 3 sentiment comparisons
            break
            
        valid_mask = (df[col1].notna()) & (df[col2].notna()) & \
                    (df[col1] != '') & (df[col2] != '')
        
        if valid_mask.sum() == 0:
            continue
            
        y1 = df.loc[valid_mask, col1]
        y2 = df.loc[valid_mask, col2]
        
        # Get unique labels
        labels = sorted(list(set(y1.unique()) | set(y2.unique())))
        
        # Create confusion matrix
        cm = confusion_matrix(y1, y2, labels=labels)
        
        # Plot
        row, col = divmod(i, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=axes[row, col])
        axes[row, col].set_title(f'Sentiment: {title}')
        axes[row, col].set_xlabel(col2.replace('sentiment_', '').upper())
        axes[row, col].set_ylabel(col1.replace('sentiment_', '').upper())
    
    # Emotion comparison
    valid_mask = (df['emotion_bertweet'].notna()) & (df['emotion_roberta'].notna()) & \
                (df['emotion_bertweet'] != 'NA') & (df['emotion_roberta'] != 'NA') & \
                (df['emotion_bertweet'] != '') & (df['emotion_roberta'] != '')
    
    if valid_mask.sum() > 0:
        y1 = df.loc[valid_mask, 'emotion_bertweet']
        y2 = df.loc[valid_mask, 'emotion_roberta']
        
        # Get unique labels
        labels = sorted(list(set(y1.unique()) | set(y2.unique())))
        
        # Create confusion matrix
        cm = confusion_matrix(y1, y2, labels=labels)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=labels, yticklabels=labels, ax=axes[1, 1])
        axes[1, 1].set_title('Emotion: BERTweet vs RoBERTa')
        axes[1, 1].set_xlabel('RoBERTa')
        axes[1, 1].set_ylabel('BERTweet')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_agreement.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_kappa_summary_table(sentiment_results, emotion_results):
    """Create a summary table of all kappa values"""
    all_results = sentiment_results + emotion_results
    
    if not all_results:
        print("❌ No results to display")
        return
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    print("\n\n📋 COHEN'S KAPPA SUMMARY TABLE:")
    print("=" * 70)
    print(f"{'Comparison':<25} {'Kappa':<8} {'Interpretation':<18} {'Samples':<8}")
    print("-" * 70)
    
    for _, row in df_results.iterrows():
        print(f"{row['Comparison']:<25} {row['Kappa']:<8.4f} {row['Interpretation']:<18} {row['Valid_Samples']:<8}")
    
    # Save to CSV
    df_results.to_csv('cohens_kappa_results.csv', index=False)
    print(f"\n💾 Results saved to: cohens_kappa_results.csv")
    
    return df_results

def main():
    """Main execution function for Task 1"""
    print("🚀 STARTING COHEN'S KAPPA ANALYSIS")
    print("=" * 55)
    
    # Load the annotated dataset
    try:
        df = pd.read_csv('annotated_reddit_posts.csv')
        print(f"✅ Dataset loaded: {len(df)} samples")
        
        # Show data overview
        print(f"\n📊 Data Overview:")
        print(f"   Total samples: {len(df)}")
        print(f"   Sentiment columns: sentiment_vader, sentiment_bertweet, sentiment_roberta")
        print(f"   Emotion columns: emotion_bertweet, emotion_roberta (emotion_vader = NA)")
        
    except FileNotFoundError:
        print("❌ Error: annotated_reddit_posts.csv not found!")
        return
    
    # Calculate kappa for sentiments
    sentiment_results = calculate_kappa_for_sentiments(df)
    
    # Calculate kappa for emotions
    emotion_results = calculate_kappa_for_emotions(df)
    
    # Create summary table
    summary_df = create_kappa_summary_table(sentiment_results, emotion_results)
    
    # Create confusion matrices
    try:
        create_confusion_matrices(df)
        print(f"\n📈 Confusion matrices saved to: confusion_matrices_agreement.png")
    except Exception as e:
        print(f"⚠️  Could not create confusion matrices: {e}")
    
    print(f"\n🎉 TASK 1 COMPLETED!")
    print(f"📋 Summary: Calculated Cohen's Kappa for {len(sentiment_results)} sentiment and {len(emotion_results)} emotion comparisons")
    
    return summary_df

if __name__ == "__main__":
    main()