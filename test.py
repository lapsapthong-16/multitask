import pandas as pd

# Read the CSV file
df = pd.read_csv('annotated_reddit_posts.csv')

# Count unique values in 'sentiment' column
sentiment_counts = df['sentiment'].value_counts()
print('Sentiment counts:')
print(sentiment_counts)
print('\n')

# Count unique values in 'emotion' column
emotion_counts = df['emotion'].value_counts()
print('Emotion counts:')
print(emotion_counts)