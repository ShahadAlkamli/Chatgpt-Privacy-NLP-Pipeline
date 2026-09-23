"""
Sentiment analysis on preprocessed ChatGPT privacy tweets.

Classifies each tweet as positive, negative, or neutral using TextBlob
polarity, then saves a bar chart and a pie chart of the distribution.

Input:  Data/preprocessed_tweets.csv
Output: figures/sentiment_distribution.png
        figures/sentiment_proportions.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Paths (relative to the repository root)
INPUT_FILE = 'Data/preprocessed_tweets.csv'
OUTPUT_DIR = 'figures'

TEXT_COLUMN = 'processed_tweet'
COLORS = ['green', 'gray', 'red']


def analyze_sentiment(tweet):
    """Classify a tweet as positive, negative, or neutral."""
    polarity = TextBlob(str(tweet)).sentiment.polarity

    if polarity > 0:
        return 'positive'
    if polarity < 0:
        return 'negative'
    return 'neutral'


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    print(f"Loaded {len(df)} tweets")

    df['Sentiment'] = df[TEXT_COLUMN].apply(analyze_sentiment)
    sentiment_counts = df['Sentiment'].value_counts()

    print('\nSentiment distribution:')
    for sentiment, count in sentiment_counts.items():
        share = count / len(df) * 100
        print(f"  {sentiment:<9} {count:>6}  ({share:.1f}%)")

    # Bar chart of sentiment counts
    plt.figure(figsize=(8, 6), dpi=300)
    sentiment_counts.plot(kind='bar', color=COLORS)
    plt.title('Sentiment Distribution of ChatGPT Privacy Tweets')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution.png'))
    plt.close()

    # Pie chart of sentiment proportions
    plt.figure(figsize=(8, 8), dpi=300)
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=COLORS
    )
    plt.title('Sentiment Proportions')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_proportions.png'))
    plt.close()

    print(f"\nFigures saved to {OUTPUT_DIR}/")
