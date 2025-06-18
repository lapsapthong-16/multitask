import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime

# Define the query for the crisis period
query = '"Galaxy Note 7" (fire OR explode OR recall) lang:en since:2016-08-01 until:2016-12-31'
max_tweets = 100000  # Define how many tweets you want to collect

# Initialize a list to store the tweets
tweets_data = []

# Collect tweets using snscrape
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets_data) < max_tweets:
        tweets_data.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.retweetCount, tweet.likeCount])
    
# Convert the list to a DataFrame for easier manipulation
df = pd.DataFrame(tweets_data, columns=['Date', 'Tweet ID', 'Text', 'Username', 'Retweets', 'Likes'])
df['Date'] = pd.to_datetime(df['Date'])

# Save the data to a CSV file
df.to_csv(f'galaxy_note7_tweets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
print(f"Scraped and saved {len(df)} tweets")