# import tweepy
# import json
# import csv
# from datetime import datetime

# # Replace these with your own credentials
# bearer_token = 'AAAAAAAAAAAAAAAAAAAAAHRM2QEAAAAA4CZj12kxESIXrH58QEJcQmtM9%2BM%3DiYLRBSIvzr0dJ8Nx9K7evSezCeOEKwTG6IzCu40Y8a6eJ6mUXB'

# # Create API v2 client
# client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# # Define the search query and the number of tweets to collect
# query = "samsung lang:en"
# max_tweets = 100

# # Create timestamp for filename
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Initialize lists to store tweet data
# tweets_data = []

# print(f"Collecting tweets for query: '{query}'...")

# try:
#     # Collect tweets using API v2 (simplified)
#     tweets = tweepy.Paginator(
#         client.search_recent_tweets,
#         query=query,
#         max_results=100,
#         limit=1
#     ).flatten(limit=max_tweets)

#     for tweet in tweets:
#         # Store basic tweet data
#         tweet_data = {
#             'id': tweet.id,
#             'text': tweet.text
#         }
#         tweets_data.append(tweet_data)
#         print(f"Collected tweet {len(tweets_data)}: {tweet.text[:50]}...")

#     print(f"\nCollected {len(tweets_data)} tweets")

#     # Save as JSON file
#     json_filename = f"tweets_{query.replace(' ', '_').replace(':', '_')}_{timestamp}.json"
#     with open(json_filename, 'w', encoding='utf-8') as f:
#         json.dump(tweets_data, f, ensure_ascii=False, indent=2)
#     print(f"Tweets saved to {json_filename}")

#     # Save as CSV file
#     csv_filename = f"tweets_{query.replace(' ', '_').replace(':', '_')}_{timestamp}.csv"
#     if tweets_data:
#         with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=tweets_data[0].keys())
#             writer.writeheader()
#             writer.writerows(tweets_data)
#         print(f"Tweets also saved to {csv_filename}")

#     print("\nDownload complete!")

# except Exception as e:
#     print(f"An error occurred: {e}")

import snscrape.modules.twitter as sntwitter
import pandas as pd

query = '"Galaxy Note 7" (fire OR explode OR recall) lang:en'
max_tweets = 100000  # Define number of tweets to collect

# Collect tweets using snscrape
tweets = []
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) < max_tweets:
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

# Create a dataframe and save to CSV
df = pd.DataFrame(tweets, columns=['Date', 'Tweet ID', 'Text', 'Username'])
df.to_csv('note7_tweets.csv', index=False)