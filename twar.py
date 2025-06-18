import twarc
import pandas as pd
from datetime import datetime
import json

def collect_tweets_with_twarc():
    """
    Collect tweets using twarc (Twitter API Academic Research Client)
    """
    
    try:
        # Initialize twarc client with API credentials
        # You'll need to get these from Twitter Developer Portal
        consumer_key = "YOUR_CONSUMER_KEY"
        consumer_secret = "YOUR_CONSUMER_SECRET"
        access_token = "YOUR_ACCESS_TOKEN"
        access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"
        
        # Create twarc client
        t = twarc.Twarc(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
        
        # Define search parameters
        query = '"Galaxy Note 7" (fire OR explode OR recall) lang:en'
        max_tweets = 1000
        
        print(f"Collecting tweets with query: '{query}'...")
        
        # Initialize list to store tweets
        tweets_data = []
        
        # Search for tweets
        count = 0
        for tweet in t.search(query):
            if count >= max_tweets:
                break
                
            tweet_data = {
                'Date': tweet['created_at'],
                'Tweet ID': tweet['id_str'],
                'Text': tweet['full_text'] if 'full_text' in tweet else tweet['text'],
                'Username': tweet['user']['screen_name'],
                'Retweets': tweet['retweet_count'],
                'Likes': tweet['favorite_count'],
                'User Followers': tweet['user']['followers_count']
            }
            tweets_data.append(tweet_data)
            count += 1
            
            if count % 100 == 0:
                print(f"Collected {count} tweets...")
        
        # Convert to DataFrame
        df = pd.DataFrame(tweets_data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'galaxy_note7_twarc_tweets_{timestamp}.csv'
        df.to_csv(filename, index=False)
        
        # Save raw JSON data as well
        json_filename = f'galaxy_note7_twarc_tweets_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(tweets_data, f, ensure_ascii=False, indent=2)
        
        print(f"Collected and saved {len(df)} tweets to {filename}")
        print(f"Raw JSON data saved to {json_filename}")
        return df
        
    except ImportError:
        print("Error: twarc library not found.")
        print("Please install twarc first:")
        print("pip install twarc")
        return None
        
    except Exception as e:
        print(f"Error collecting tweets: {e}")
        print("\nMake sure you have:")
        print("1. Valid Twitter API credentials")
        print("2. Proper API access (Academic Research or Elevated)")
        return None

def setup_twarc_config():
    """
    Help setup twarc configuration
    """
    print("To configure twarc with your Twitter API credentials:")
    print("1. Get credentials from https://developer.twitter.com/")
    print("2. Run: twarc2 configure")
    print("3. Or edit the credentials in this script")
    print("\nFor twarc v1:")
    print("Run: twarc configure")

def collect_with_twarc2():
    """
    Alternative using twarc2 (newer version)
    """
    try:
        from twarc import Twarc2
        
        # Initialize twarc2 client
        # Bearer token method (simpler)
        bearer_token = "YOUR_BEARER_TOKEN"
        t = Twarc2(bearer_token=bearer_token)
        
        query = '"Galaxy Note 7" (fire OR explode OR recall) lang:en'
        
        print(f"Collecting tweets with twarc2...")
        tweets_data = []
        
        # Search recent tweets (last 7 days)
        for response in t.search_recent(query, max_results=100):
            for tweet in response['data']:
                tweet_data = {
                    'Date': tweet['created_at'],
                    'Tweet ID': tweet['id'],
                    'Text': tweet['text'],
                    'Author ID': tweet['author_id'],
                    'Public Metrics': tweet.get('public_metrics', {})
                }
                tweets_data.append(tweet_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(tweets_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'galaxy_note7_twarc2_tweets_{timestamp}.csv'
        df.to_csv(filename, index=False)
        
        print(f"Collected {len(df)} tweets with twarc2")
        return df
        
    except ImportError:
        print("twarc2 not found. Install with: pip install twarc")
        return None

if __name__ == "__main__":
    print("Twitter Academic Research Client (twarc) Tweet Collector")
    print("=" * 60)
    
    # Check if twarc is available
    try:
        import twarc
        print("twarc library found!")
        
        # Try twarc2 first (newer version)
        try:
            from twarc import Twarc2
            print("Using twarc2 (recommended)")
            df = collect_with_twarc2()
        except ImportError:
            print("Using twarc v1")
            df = collect_tweets_with_twarc()
        
        if df is not None:
            print(f"\nDataFrame shape: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())
            
    except ImportError:
        print("twarc library not found.")
        print("\nTo install twarc:")
        print("pip install twarc")
        print("\nAfter installation, configure with your API credentials:")
        setup_twarc_config()
