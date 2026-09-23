"""
LDA topic modeling on preprocessed ChatGPT privacy tweets.

Builds an LDA model with Gensim, prints the extracted topics, and
visualizes the results as a word cloud and a topic distribution chart.

The optimal number of topics (3) was determined using coherence and
perplexity scores, as reported in the published study.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud

# Paths (relative to the repository root)
INPUT_FILE = 'Data/preprocessed_tweets.csv'

NUM_TOPICS = 3
NUM_WORDS = 10
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

# Train the LDA model
lda_model = LdaModel(
    corpus,
    num_topics=NUM_TOPICS,
    id2word=dictionary,
    passes=PASSES,
    random_state=RANDOM_STATE
)

# Print extracted topics
print(f"\nExtracted {NUM_TOPICS} topics:\n")
for topic_num, words in lda_model.show_topics(num_words=NUM_WORDS, formatted=False):
    terms = ', '.join(word for word, _ in words)
    print(f"Topic {topic_num}: {terms}\n")

# Word cloud of the top terms across all topics
top_words = [
    word
    for topic_id in range(NUM_TOPICS)
    for word, _ in lda_model.show_topic(topic_id, topn=NUM_WORDS)
]

wordcloud = WordCloud(
    width=800, height=400, background_color='white'
).generate(' '.join(top_words))

plt.figure(figsize=(10, 5), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()

# Distribution of dominant topics across documents
dominant_topics = [
    max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus
]
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()

plt.figure(figsize=(10, 6), dpi=300)
sns.barplot(
    x=topic_counts.index,
    y=topic_counts.values,
    hue=topic_counts.index,
    palette='viridis',
    legend=False
)
plt.title('Dominant Topic Distribution')
plt.xlabel('Topic')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.show()

print('\nTweets per topic:')
for topic_id, count in topic_counts.items():
    print(f"  Topic {topic_id}: {count}")    topic_words = lda_model.show_topic(topic_id, topn=10)
    words = [word for word, _ in topic_words if word in dictionary.token2id]
    topics.append(words)

# Flatten the list of words
all_words = [word for sublist in topics for word in sublist]

# Concatenate all words into a single string
text = ' '.join(all_words)

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the generated image using Matplotlib with increased resolution (dpi)
plt.figure(figsize=(10, 5), dpi=300)  # Adjust the dpi parameter
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Visualize topics distribution
# Get topic distribution for each document
topics_distribution = [lda_model[doc] for doc in corpus]

# Extract the dominant topic for each document
dominant_topics = [max(topic, key=lambda x: x[1])[0] for topic in topics_distribution]

# Count the occurrences of each dominant topic
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()

# Visualize the distribution of dominant topics using a bar chart
plt.figure(figsize=(10, 6), dpi=300)  # Adjust the dpi parameter
sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
plt.title('Dominant Topic Distribution')
plt.xlabel('Dominant Topic')
plt.ylabel('Number of Tweets')
plt.show()
