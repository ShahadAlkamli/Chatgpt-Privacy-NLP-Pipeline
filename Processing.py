"""
Preprocesses the raw ChatGPT tweet dataset to isolate privacy-related
discussions.

Applies lowercasing, hyperlink and mention removal, tokenization,
stopword removal, and keyword matching, following the pipeline
described in the published study.

Input:  Data/ChatGPTtweets.csv
Output: Data/preprocessed_tweets.csv
"""

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Paths (relative to the repository root)
INPUT_FILE = 'Data/ChatGPTtweets.csv'
OUTPUT_FILE = 'Data/preprocessed_tweets.csv'

# Privacy and security keywords used to filter relevant tweets
RELEVANT_KEYWORDS = [
    'security', 'privacy', 'cybersecurity', 'confidentiality',
    'secure', 'hack', 'hacker', 'encryption', 'theft'
]

STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """Clean a tweet and return it only if privacy-related.

    Returns (original, processed) for relevant tweets, or (None, None)
    for tweets that do not mention privacy or security.
    """
    if pd.isna(text):
        return None, None

    # Lowercase, then strip hyperlinks and mentions
    cleaned = text.lower()
    cleaned = re.sub(r'http\S+', '', cleaned)
    cleaned = re.sub(r'@\w+', '', cleaned)

    # Tokenize and remove stopwords and non-alphabetic tokens
    tokens = word_tokenize(cleaned)
    tokens = [w for w in tokens if w.isalpha() and w not in STOP_WORDS]
    processed_text = ' '.join(tokens)

    # Keep only tweets matching a privacy or security keyword
    if any(keyword in processed_text for keyword in RELEVANT_KEYWORDS):
        return text, processed_text

    return None, None


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} tweets")

    df['original_tweet'], df['processed_tweet'] = zip(
        *df['content'].apply(preprocess_text)
    )

    df = df.dropna(subset=['original_tweet', 'processed_tweet'])
    print(f"Retained {len(df)} privacy-related tweets")

    df.to_csv(
        OUTPUT_FILE,
        columns=['original_tweet', 'processed_tweet'],
        index=False
    )
    print(f"Saved to {OUTPUT_FILE}")
