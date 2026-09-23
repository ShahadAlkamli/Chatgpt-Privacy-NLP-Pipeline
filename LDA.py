"""
Determines the optimal number of LDA topics for the ChatGPT privacy
tweets by evaluating coherence and perplexity across candidate values.

Reproduces the topic selection analysis reported in the published study,
which identified 3 as the optimal number of topics.

Input: Data/preprocessed_tweets.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

# Paths (relative to the repository root)
INPUT_FILE = 'Data/preprocessed_tweets.csv'

# Range of topic counts to evaluate
START = 2
LIMIT = 15
STEP = 1

PASSES = 15
RANDOM_STATE = 42

# Load preprocessed tweets
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['processed_tweet'])
print(f"Loaded {len(df)} tweets")

# Build dictionary and document-term matrix
tokenized = [text.split() for text in df['processed_tweet']]
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

# Reconstruct token lists for coherence evaluation
texts = [[dictionary[word_id] for word_id, _ in doc] for doc in corpus]

# Evaluate each candidate number of topics
coherence_scores = []
perplexity_scores = []
topic_range = range(START, LIMIT, STEP)

for num_topics in topic_range:
    lda_model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=PASSES,
        random_state=RANDOM_STATE
    )

    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence='u_mass'
    )

    coherence = coherence_model.get_coherence()
    perplexity = lda_model.log_perplexity(corpus)

    coherence_scores.append(coherence)
    perplexity_scores.append(perplexity)

    print(f"Topics: {num_topics:>2}  Coherence: {coherence:.4f}  Perplexity: {perplexity:.4f}")

# Plot coherence and perplexity against the number of topics
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Number of Topics')
ax1.set_ylabel('Coherence Score', color='tab:red')
ax1.plot(topic_range, coherence_scores, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Perplexity Score', color='tab:blue')
ax2.plot(topic_range, perplexity_scores, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Coherence and Perplexity Scores vs. Number of Topics')
plt.tight_layout()
plt.show()
