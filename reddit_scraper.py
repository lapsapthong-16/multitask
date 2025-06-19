import praw
import json
import pandas as pd
from datetime import datetime
from langdetect import detect
import time

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

# Define time range (August 1, 2016 to December 31, 2016)
start_date = datetime(2016, 8, 1).timestamp()
end_date = datetime(2016, 12, 31, 23, 59, 59).timestamp()

# Create timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize lists to store post data
posts_data = []
processed_post_ids = set()  # To avoid duplicates across queries

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
                # Skip if we've already processed this post
                if post.id in processed_post_ids:
                    continue
                
                # Check if post meets filtering criteria
                is_valid, reason = is_valid_post(post)
                
                if is_valid:
                    # Store post data
                    post_data = {
                        'id': post.id,
                        'title': post.title if post.title else '',
                        'selftext': post.selftext if post.selftext else '',
                        'author': str(post.author) if post.author else '[deleted]',
                        'created_utc': post.created_utc,
                        'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'subreddit': str(post.subreddit),
                        'search_query': search_query
                    }
                    posts_data.append(post_data)
                    processed_post_ids.add(post.id)
                    query_posts_collected += 1
                    
                    # Print progress (show title, truncated if too long)
                    title_preview = post.title[:50] + "..." if len(post.title) > 50 else post.title
                    print(f"  Collected post {len(posts_data)}: {title_preview}")
                
                # Add small delay to be respectful to Reddit's API
                time.sleep(0.1)
            
            print(f"  Query '{search_query}' yielded {query_posts_collected} valid posts")
            
        except Exception as query_error:
            print(f"  Error processing query '{search_query}': {query_error}")
            continue

    print(f"\nTotal collected: {len(posts_data)} posts")

    if posts_data:
        # Create base filename
        base_filename = f"reddit_posts_note7_crisis_{timestamp}"
        
        # Save as JSON file
        json_filename = f"{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        print(f"Posts saved to {json_filename}")
        
        # Save as CSV file using pandas
        csv_filename = f"{base_filename}.csv"
        df = pd.DataFrame(posts_data)
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"Posts saved to {csv_filename}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total posts collected: {len(posts_data)}")
        print(f"- Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        print(f"- Average score: {df['score'].mean():.1f}")
        print(f"- Posts by query:")
        for query in search_queries:
            count = len(df[df['search_query'] == query])
            print(f"  '{query}': {count} posts")
    else:
        print("No posts met the filtering criteria.")

    print("\nDownload complete!")

except Exception as e:
    print(f"An error occurred: {e}")
    if posts_data:
        # Save partial results if any were collected
        json_filename = f"reddit_posts_partial_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        print(f"Partial results saved to {json_filename}")
        
        # Also save partial results as CSV
        try:
            csv_filename = f"reddit_posts_partial_{timestamp}.csv"
            df = pd.DataFrame(posts_data)
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"Partial results saved to {csv_filename}")
        except Exception as csv_error:
            print(f"Could not save partial CSV: {csv_error}") 