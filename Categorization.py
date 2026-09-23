"""
Categorizes privacy-related tweets into three concern categories using
keyword matching, as described in the published study.

This step is independent of the LDA topic modeling and follows the
keyword lists reported in the paper.

Input:  Data/preprocessed_tweets.csv
Output: Data/categorized_tweets.csv
"""

import pandas as pd

# Paths (relative to the repository root)
INPUT_FILE = 'Data/preprocessed_tweets.csv'
OUTPUT_FILE = 'Data/categorized_tweets.csv'

# Keyword lists for each privacy concern category
CATEGORY_KEYWORDS = {
    'public_data_exploitation': [
        'public data', 'open data', 'sanitizing', 'training process',
        'training data', 'trained on', 'publicly available data'
    ],
    'personal_input_exploitation': [
        'conversation', 'conversations', 'input data', 'private chat',
        'prompt', 'prompts'
    ],
    'unauthorized_access_to_data': [
        'unauthorized access', 'vulnerabilities', 'data breach',
        'data security', 'security breach', 'attack', 'attacking',
        'hack', 'hacking', 'threat', 'breach', 'unauthorized entry',
        'data compromise'
    ],
}


def matches(tweet, keywords):
    """Check whether a tweet contains any of the given keywords."""
    return isinstance(tweet, str) and any(
        keyword in tweet.lower() for keyword in keywords
    )


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} tweets")

    # Build one data frame per category, then combine them
    frames = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        tweets = [t for t in df['original_tweet'] if matches(t, keywords)]
        frames.append(pd.DataFrame({'content': tweets, 'category': category}))
        print(f"  {category}: {len(tweets)}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(combined)} categorized tweets to {OUTPUT_FILE}")
