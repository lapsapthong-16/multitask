import praw
import json
import pandas as pd
from datetime import datetime
from langdetect import detect
from textblob import TextBlob
import time
import re
import os  # Added for file existence check

# Replace these with your own Reddit API credentials
client_id = 'zwNVQTjvLRlJBm4IytY5nA'
client_secret = 'OtY1GFZNIpqep-2UoiU8qiMQmSwhhg'
user_agent = "linux:note7_sentiment_scraper:1.0 (by /u/MinimumBeginning2278)"

# Create Reddit API client
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Define the search parameters
subreddit_name = 'samsung'
search_queries = ["Galaxy Note 7", "Note7", "Samsung fire", "Note 7 recall", "#Note7Recall", "#SamsungFire"]
max_posts_per_query = 100
max_comments_per_post = 10

# Define time range (August 1, 2016 to December 31, 2016)
start_date = datetime(2016, 8, 1).timestamp()
end_date = datetime(2016, 12, 31, 23, 59, 59).timestamp()

# Create timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize lists to store post data
posts_data = []
processed_post_ids = set()  # To avoid duplicates across queries

# File to track collected IDs
collected_ids_file = "collected_ids.json"  # Added: file to store previously collected IDs

# Load previously collected IDs from file
def load_collected_ids():
    """Load previously collected post and comment IDs from file"""
    if os.path.exists(collected_ids_file):
        try:
            with open(collected_ids_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('post_ids', [])), set(data.get('comment_ids', []))
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: Could not load {collected_ids_file}, starting fresh")
            return set(), set()
    else:
        print(f"{collected_ids_file} not found, starting fresh")
        return set(), set()

# Save collected IDs to file
def save_collected_ids(post_ids, comment_ids):
    """Save collected post and comment IDs to file"""
    data = {
        'post_ids': list(post_ids),
        'comment_ids': list(comment_ids),
        'last_updated': datetime.now().isoformat()
    }
    with open(collected_ids_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Load existing collected IDs at startup
existing_post_ids, existing_comment_ids = load_collected_ids()
print(f"Loaded {len(existing_post_ids)} previously collected post IDs and {len(existing_comment_ids)} comment IDs")

# News domains for classification
news_domains = [
    'cnn', 'bbc', 'reuters', 'nytimes', 'washingtonpost', 'theguardian', 
    'forbes', 'bloomberg', 'techcrunch', 'verge', 'engadget', 'androidpolice',
    'androidcentral', 'gsmarena', 'phonearena', '9to5google', 'arstechnica'
]

print(f"Collecting posts from r/{subreddit_name} for multiple queries...")
print(f"Time range: August 1, 2016 - December 31, 2016")
print(f"Queries: {search_queries}")

def is_english(text):
    """Check if text is in English using langdetect"""
    try:
        if not text or len(text.strip()) < 3:
            return False
        return detect(text) == 'en'
    except:
        return False

def get_subjectivity(text):
    """Get subjectivity score using TextBlob, handle errors gracefully"""
    try:
        if not text or len(text.strip()) < 3:
            return 0.0
        blob = TextBlob(text)
        return blob.sentiment.subjectivity
    except:
        return 0.0

def classify_post_type(post):
    """Classify post as 'news' or 'opinion' based on URL and content"""
    try:
        # Check if URL contains news domain
        for domain in news_domains:
            if domain.lower() in post.url.lower():
                return "news"
        
        # Check if it's a self post (selftext exists and URL is Reddit permalink)
        if post.selftext and post.selftext.strip() and 'reddit.com' in post.url:
            return "opinion"
        
        # Default classification
        return "other"
    except:
        return "unknown"

def extract_comments(post, max_comments=10):
    """Extract top-level comments from a post"""
    comments_data = []
    try:
        # Load all comments
        post.comments.replace_more(limit=0)
        
        # Get top-level comments only
        top_comments = post.comments[:max_comments]
        
        for comment in top_comments:
            try:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    # Check if comment ID already exists in collected IDs
                    if comment.id in existing_comment_ids:
                        print(f"    Skipping already collected comment: {comment.id}")
                        continue  # Skip already collected comment
                    
                    comment_text = comment.body if comment.body else ''
                    comment_subjectivity = get_subjectivity(comment_text)
                    
                    # Apply subjectivity filter to comments too (optional)
                    if comment_subjectivity >= 0.4:
                        comment_data = {
                            'comment_id': comment.id,
                            'comment_text': comment_text,
                            'comment_score': comment.score if hasattr(comment, 'score') else 0,
                            'comment_author': str(comment.author) if comment.author else '[deleted]',
                            'comment_subjectivity': comment_subjectivity
                        }
                        comments_data.append(comment_data)
                        # Add comment ID to existing set for this session
                        existing_comment_ids.add(comment.id)
            except Exception as comment_error:
                continue  # Skip problematic comments
                
    except Exception as e:
        print(f"    Error extracting comments: {e}")
    
    return comments_data

def is_valid_post(post):
    """Check if post meets all filtering criteria"""
    # Time filter
    if not (start_date <= post.created_utc <= end_date):
        return False, "time_range"
    
    # Score filter
    if post.score < 2:
        return False, "low_score"
    
    # Content filter - check if both title and selftext are empty/missing
    title = post.title.strip() if post.title else ""
    selftext = post.selftext.strip() if post.selftext else ""
    
    if not title and not selftext:
        return False, "empty_content"
    
    # Language filter
    combined_text = f"{title} {selftext}".strip()
    if not is_english(combined_text):
        return False, "non_english"
    
    # Subjectivity filter
    subjectivity_score = get_subjectivity(combined_text)
    if subjectivity_score < 0.4:
        return False, "low_subjectivity"
    
    return True, "valid"

try:
    # Get the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    
    # Search for each query
    for query_idx, search_query in enumerate(search_queries):
        print(f"\n--- Processing query {query_idx + 1}/{len(search_queries)}: '{search_query}' ---")
        
        try:
            # Search for posts in the subreddit
            search_results = subreddit.search(search_query, limit=max_posts_per_query)
            
            query_posts_collected = 0
            for post in search_results:
                # Skip if we've already processed this post in this session
                if post.id in processed_post_ids:
                    continue
                
                # Check if post ID already exists in collected IDs from previous runs
                if post.id in existing_post_ids:
                    print(f"  Skipping already collected post: {post.id}")
                    processed_post_ids.add(post.id)  # Add to session set to avoid reprocessing
                    continue
                
                # Check if post meets filtering criteria
                is_valid, reason = is_valid_post(post)
                
                if is_valid:
                    # Get post content for analysis
                    title = post.title if post.title else ''
                    selftext = post.selftext if post.selftext else ''
                    combined_text = f"{title} {selftext}".strip()
                    
                    # Get subjectivity and post type
                    subjectivity_score = get_subjectivity(combined_text)
                    post_type = classify_post_type(post)
                    
                    print(f"  Extracting comments for post: {title[:30]}...")
                    
                    # Extract comments (already handles comment ID checking internally)
                    comments = extract_comments(post, max_comments_per_post)
                    
                    # Store post data
                    post_data = {
                        'id': post.id,
                        'title': title,
                        'selftext': selftext,
                        'author': str(post.author) if post.author else '[deleted]',
                        'created_utc': post.created_utc,
                        'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'subreddit': str(post.subreddit),
                        'search_query': search_query,
                        'subjectivity': round(subjectivity_score, 3),
                        'post_type': post_type,
                        'extracted_comments': comments,
                        'num_extracted_comments': len(comments)
                    }
                    posts_data.append(post_data)
                    processed_post_ids.add(post.id)
                    # Add post ID to existing set for this session
                    existing_post_ids.add(post.id)
                    query_posts_collected += 1
                    
                    # Print progress (show title, truncated if too long)
                    title_preview = title[:50] + "..." if len(title) > 50 else title
                    print(f"  Collected post {len(posts_data)}: {title_preview} (subj: {subjectivity_score:.2f}, type: {post_type}, comments: {len(comments)})")
                else:
                    if reason == "low_subjectivity":
                        print(f"  Skipped post (low subjectivity): {post.title[:30]}...")
                
                # Add small delay to be respectful to Reddit's API
                time.sleep(0.2)  # Increased delay due to comment extraction
            
            print(f"  Query '{search_query}' yielded {query_posts_collected} valid posts")
            
        except Exception as query_error:
            print(f"  Error processing query '{search_query}': {query_error}")
            continue

    print(f"\nTotal collected: {len(posts_data)} posts")

    if posts_data:
        # Create base filename
        base_filename = f"reddit_posts_note7_enhanced_{timestamp}"
        
        # Save as JSON file
        json_filename = f"{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        print(f"Posts saved to {json_filename}")
        
        # Create DataFrame for summary statistics (only if posts_data is not empty)
        df = pd.DataFrame(posts_data)
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total posts collected: {len(posts_data)}")
        print(f"- Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        print(f"- Average score: {df['score'].mean():.1f}")
        print(f"- Average subjectivity: {df['subjectivity'].mean():.3f}")
        print(f"- Total comments extracted: {df['num_extracted_comments'].sum()}")
        print(f"- Post types distribution:")
        post_type_counts = df['post_type'].value_counts()
        for post_type, count in post_type_counts.items():
            print(f"  {post_type}: {count} posts")
        print(f"- Posts by query:")
        for query in search_queries:
            count = len(df[df['search_query'] == query])
            print(f"  '{query}': {count} posts")
    else:
        print("No new posts met the filtering criteria.")

    # Update collected_ids.json with all IDs (both existing and new)
    save_collected_ids(existing_post_ids, existing_comment_ids)
    print(f"Updated {collected_ids_file} with {len(existing_post_ids)} post IDs and {len(existing_comment_ids)} comment IDs")

    print("\nDownload complete!")

except Exception as e:
    print(f"An error occurred: {e}")
    if posts_data:
        # Save partial results if any were collected
        json_filename = f"reddit_posts_partial_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        print(f"Partial results saved to {json_filename}")
    
    # Still update collected IDs even if there was an error
    save_collected_ids(existing_post_ids, existing_comment_ids)
    print(f"Updated {collected_ids_file} with current IDs")
        