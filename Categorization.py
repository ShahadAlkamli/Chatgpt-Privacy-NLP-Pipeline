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
    'Public Data Exploitation': [
        'public data', 'open data', 'sanitizing', 'training process',
        'training data', 'trained on', 'publicly available data'
    ],
    'Personal Input Exploitation': [
        'conversation', 'conversations', 'input data', 'private chat',
        'prompt', 'prompts'
    ],
    'Unauthorized Access to Data': [
        'unauthorized access', 'vulnerabilities', 'data breach',
        'data security', 'security breach', 'attack', 'attacking',
        'hack', 'hacking', 'threat', 'breach', 'unauthorized entry',
        'data compromise'
    ],
}


def categorize(tweet):
    """Return every category whose keywords appear in the tweet."""
    if not isinstance(tweet, str):
        return []

    lowered = tweet.lower()
    return [
        category
        for category, keywords in CATEGORY_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} tweets")

    # Assign each tweet to one row per matching category
    rows = []
    for tweet in df['original_tweet']:
        for category in categorize(tweet):
            rows.append({'category': category, 'tweet': tweet})

    categorized = pd.DataFrame(rows)

    print('\nTweets per category:')
    for category in CATEGORY_KEYWORDS:
        count = (categorized['category'] == category).sum()
        print(f"  {category}: {count}")

    categorized.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
