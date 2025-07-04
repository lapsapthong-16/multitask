import json
import pandas as pd
import re
import string
# import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class RedditTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Remove some emotion-related words from stopwords to preserve sentiment
        emotion_words = {
            'not', 'no', 'nor', 'but', 'however', 'although', 'though',
            'very', 'really', 'quite', 'too', 'so', 'more', 'most',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very'
        }
        self.stop_words = self.stop_words - emotion_words
        
        print(f"Initialized preprocessor with {len(self.stop_words)} stopwords")
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        # Remove http/https URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove www URLs
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove reddit links like /r/subreddit
        text = re.sub(r'/r/[A-Za-z0-9_]+', '', text)
        return text
    
    def remove_mentions_hashtags(self, text):
        """Remove @mentions and #hashtags"""
        # Remove @mentions
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        # Remove #hashtags but preserve the word (e.g., #BombsAway -> BombsAway)
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
        return text
    
    def remove_html_tags(self, text):
        """Remove HTML tags"""
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def handle_reddit_formatting(self, text):
        """Handle Reddit-specific formatting"""
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove reddit user references like u/username
        text = re.sub(r'u/[A-Za-z0-9_]+', '', text)
        # Remove markdown formatting **bold** and *italic*
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        # Remove quote markers
        text = re.sub(r'^>', '', text, flags=re.MULTILINE)
        return text
    
    def convert_emojis(self, text):
        """Convert emojis to text descriptions"""
        try:
            import emoji
            # Convert emojis to text
            text = emoji.demojize(text, delimiters=(" ", " "))
            # Clean up the emoji text formatting
            text = re.sub(r':[a-zA-Z_]+:', lambda m: m.group().replace('_', ' ').replace(':', ''), text)
        except ImportError:
            # If emoji package is not available, just return the text as is
            pass
        return text
    
    def clean_special_characters(self, text):
        """Remove special characters but preserve emotionally relevant punctuation"""
        # Preserve ! and ? as they convey emotion
        # First, protect exclamation and question marks
        text = re.sub(r'!+', ' EXCLAMATION ', text)
        text = re.sub(r'\?+', ' QUESTION ', text)
        
        # Remove other punctuation except apostrophes (for contractions)
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Restore exclamation and question marks
        text = text.replace(' EXCLAMATION ', ' ! ')
        text = text.replace(' QUESTION ', ' ? ')
        
        # Handle contractions by removing apostrophes after processing
        text = re.sub(r"'", '', text)
        
        return text
    
    def normalize_whitespace(self, text):
        """Remove redundant whitespace"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords while preserving sentence structure"""
        try:
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
        except:
            # Fallback: simple split if tokenization fails
            words = text.lower().split()
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """Lemmatize words to their base form"""
        try:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except:
            # Fallback: simple split if tokenization fails
            words = text.split()
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
    
    def anonymize_identifiers(self, text):
        """Remove or mask identifiable information"""
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # Remove potential usernames (sequences of letters/numbers/underscores)
        text = re.sub(r'\b[A-Za-z0-9_]{8,}\b', '[USERNAME]', text)
        return text
    
    def preprocess_text(self, text):
        """Apply all preprocessing steps"""
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Step 1: Handle Reddit-specific formatting
        text = self.handle_reddit_formatting(text)
        
        # Step 2: Remove URLs
        text = self.remove_urls(text)
        
        # Step 3: Remove mentions and hashtags
        text = self.remove_mentions_hashtags(text)
        
        # Step 4: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 5: Convert emojis
        text = self.convert_emojis(text)
        
        # Step 6: Anonymize identifiers
        text = self.anonymize_identifiers(text)
        
        # Step 7: Clean special characters (preserve ! and ?)
        text = self.clean_special_characters(text)
        
        # Step 8: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 9: Convert to lowercase
        text = text.lower()
        
        # Step 10: Remove stopwords
        text = self.remove_stopwords(text)
        
        # Step 11: Lemmatize
        text = self.lemmatize_text(text)
        
        # Final cleanup
        text = self.normalize_whitespace(text)
        
        return text

def load_and_preprocess_data():
    """Load the Reddit data and apply preprocessing"""
    
    print("Loading Reddit posts data...")
    
    # Load the JSON data
    try:
        with open('combined_y_labeled_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Loaded {len(data)} Reddit posts/comments")
    except FileNotFoundError:
        print("Error: combined_y_labeled_data.json not found!")
        return
    
    # Initialize preprocessor
    preprocessor = RedditTextPreprocessor()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("Starting text preprocessing...")
    print("This may take a few minutes...")
    
    # Apply preprocessing
    processed_texts = []
    for i, text in enumerate(df['text_content']):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(df)} texts...")
        
        processed_text = preprocessor.preprocess_text(text)
        processed_texts.append(processed_text)
    
    # Create final dataset
    final_df = pd.DataFrame({
        'id': df['id'],
        'text_content': processed_texts,
        'original_text': df['text_content'],
        'type': df['type'],
        'score': df['score'],
        'subjectivity': df['subjectivity']
    })
    
    # Remove entries where cleaned text is empty or too short
    final_df = final_df[final_df['text_content'].str.len() >= 3]
    
    print(f"\nPreprocessing complete!")
    print(f"Original dataset: {len(df)} entries")
    print(f"Final dataset: {len(final_df)} entries")
    print(f"Removed {len(df) - len(final_df)} entries with insufficient content")
    
    # Save to CSV
    final_df.to_csv('cleaned_reddit_posts.csv', index=False, encoding='utf-8')
    print(f"\nCleaned dataset saved to: cleaned_reddit_posts.csv")
    
    # Display sample results
    print("\n" + "="*60)
    print("SAMPLE PREPROCESSING RESULTS")
    print("="*60)
    
    for i in range(min(3, len(final_df))):
        print(f"\nSample {i+1}:")
        print(f"Original: {final_df.iloc[i]['original_text'][:150]}...")
        print(f"Cleaned:  {final_df.iloc[i]['text_content'][:150]}...")
        print("-" * 40)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total entries: {len(final_df)}")
    print(f"Posts: {len(final_df[final_df['type'] == 'post'])}")
    print(f"Comments: {len(final_df[final_df['type'] == 'comment'])}")
    
    # Text length statistics
    text_lengths = final_df['text_content'].str.len()
    print(f"\nText length statistics (after cleaning):")
    print(f"  Average: {text_lengths.mean():.1f} characters")
    print(f"  Median: {text_lengths.median():.1f} characters")
    print(f"  Min: {text_lengths.min()} characters")
    print(f"  Max: {text_lengths.max()} characters")
    
    # Word count statistics
    word_counts = final_df['text_content'].str.split().str.len()
    print(f"\nWord count statistics (after cleaning):")
    print(f"  Average: {word_counts.mean():.1f} words")
    print(f"  Median: {word_counts.median():.1f} words")
    print(f"  Min: {word_counts.min()} words")
    print(f"  Max: {word_counts.max()} words")
    
    # Score and subjectivity statistics
    print(f"\nScore statistics:")
    print(f"  Average: {final_df['score'].mean():.2f}")
    print(f"  Range: {final_df['score'].min()} to {final_df['score'].max()}")
    
    print(f"\nSubjectivity statistics:")
    print(f"  Average: {final_df['subjectivity'].mean():.3f}")
    print(f"  Range: {final_df['subjectivity'].min():.3f} to {final_df['subjectivity'].max():.3f}")
    
    return final_df

if __name__ == "__main__":
    print("Reddit Posts Text Preprocessing")
    print("=" * 50)
    
    # Install required packages if not available
    try:
        import emoji
    except ImportError:
        print("Installing emoji package...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'emoji'])
        import emoji
    
    # Run preprocessing
    cleaned_data = load_and_preprocess_data()
    
    if cleaned_data is not None:
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*60)
        print("Your dataset is now ready for sentiment and emotion analysis.")
        print("Output file: cleaned_reddit_posts.csv")
        print("\nNext steps:")
        print("1. Review the cleaned data")
        print("2. Apply sentiment analysis models")
        print("3. Apply emotion classification models")
        print("4. Analyze results for PR crisis insights")