# ChatGPT Privacy Analysis – NLP Pipeline

### Based on the research paper:

**"Understanding privacy concerns in ChatGPT: A data-driven approach with LDA topic modeling" (Heliyon, 2024)**
Authors: Shahad Alkamli & Reham Alabduljabbar

---

## 📌 Overview

This repository contains the Natural Language Processing (NLP) pipeline used in the research study
*Understanding privacy concerns in ChatGPT: A data-driven approach with LDA topic modeling*.

The study analyzes tweets discussing ChatGPT to identify user privacy concerns through:

- Data preprocessing and keyword-based filtering
- LDA topic modeling for thematic exploration
- Sentiment analysis
- Keyword-driven categorization into privacy concern categories

---

## 📁 Repository Structure

```
├── Data/
│     └── preprocessed_tweets.csv     # The 11k processed tweets used in the analysis
│
├── Processing.py                     # Preprocessing: cleaning, tokenization, keyword filtering
├── LDA.py                            # Topic modeling and optimal topic count selection
├── Categorization.py                 # Keyword-driven categorization into privacy categories
├── Sentiment.py                      # Sentiment analysis and visualization
└── README.md
```

The included dataset matches the refined dataset described in the study: 500k raw tweets reduced to 11k privacy-related tweets.

---

## 🔬 Methodology

### **1. Data Preprocessing**

`Processing.py` refines the raw dataset of 500k tweets mentioning ChatGPT:

- Converts text to lowercase
- Removes hyperlinks and mentions
- Tokenizes with NLTK
- Removes English stopwords
- Filters by privacy keywords: *security, privacy, cybersecurity, confidentiality, secure, hack, hacker, encryption, theft*

This produces the 11k refined tweets used throughout the analysis.

### **2. Topic Modeling**

`LDA.py` uses **Gensim** to evaluate candidate topic counts from 2 to 14, computing coherence (u_mass) and perplexity for each. Based on these metrics, the optimal number of topics was determined to be **3**.

Topic modeling provides a high-level overview of discussion themes and is independent of the categorization step below.

### **3. Sentiment Analysis**

`Sentiment.py` uses **TextBlob** to classify each tweet as positive, negative, or neutral based on polarity, then visualizes the distribution as a bar chart and a pie chart.

### **4. Data Categorization**

`Categorization.py` classifies tweets into three privacy concern categories using keyword matching, independently of the topic model:

| Category | Keywords |
|----------|----------|
| **Public Data Exploitation** | public data, open data, sanitizing, training process, training data, trained on, publicly available data |
| **Personal Input Exploitation** | conversation, conversations, input data, private chat, prompt, prompts |
| **Unauthorized Access to Data** | unauthorized access, vulnerabilities, data breach, data security, security breach, attack, attacking, hack, hacking, threat, breach, unauthorized entry, data compromise |

---

## ▶️ Running the Project

### Install dependencies

```bash
pip install numpy pandas nltk gensim textblob matplotlib
```

Download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run the pipeline

```bash
python Processing.py        # preprocessing and filtering
python LDA.py               # topic modeling
python Sentiment.py         # sentiment analysis
python Categorization.py    # privacy categorization
```

`Processing.py` expects the raw dataset at `Data/ChatGPTtweets.csv`. The preprocessed output is already included, so the remaining scripts can be run directly.

---

## 📊 Dataset

`Data/preprocessed_tweets.csv` contains:

- 11k tweets filtered for privacy discussions
- Original and processed text for each tweet
- No usernames or metadata, in compliance with Twitter's terms of service

The raw dataset of 500k ChatGPT tweets is available on [Kaggle](https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023).

---

## 📚 Citation

If you use this code or dataset, please cite:

```
Alkamli, S., & Alabduljabbar, R. (2024). Understanding privacy concerns in
ChatGPT: A data-driven approach with LDA topic modeling.
Heliyon, 10(20), e39087. https://doi.org/10.1016/j.heliyon.2024.e39087
```

---

## 📝 License

This repository is provided for academic and research purposes.
