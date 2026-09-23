# ChatGPT Privacy Analysis – NLP Pipeline

### Based on the research paper:

**"Understanding privacy concerns in ChatGPT: A data-driven approach with LDA topic modeling" (Heliyon, 2024)**
Authors: Shahad Alkamli & Reham Alabduljabbar

---

## 📌 Overview

This repository contains the full Natural Language Processing (NLP) pipeline used in the research study
*Understanding privacy concerns in ChatGPT: A data-driven approach with LDA topic modeling*.
The project analyzes tweets discussing ChatGPT to identify user privacy concerns using:

- Data preprocessing
- Tokenization and segmentation
- Latent Dirichlet Allocation (LDA) topic modeling
- Sentiment analysis
- Keyword-based categorization into privacy categories

This work supports the findings presented in the published study.

---

## 📁 Repository Structure

```
├── Data/
│     └── preprocessed_tweets.csv     # The 11k processed tweets used in the analysis
│
├── Processing.py                     # Preprocessing: cleaning, tokenization, stopword removal
├── Segmentation.py                   # Additional segmentation and tokenization utilities
├── LDA.py                            # LDA topic modeling with Gensim
├── Sentiment.py                      # Sentiment analysis module
└── README.md
```

The included dataset matches the refined dataset described in the research study
(processed from 500k tweets → 11k privacy-related tweets).

---

## 🔬 Methodology Summary

### **1. Data Preprocessing**

Performed using `Processing.py`, which:

- Converts text to lowercase
- Removes links and mentions
- Tokenizes words
- Removes stopwords
- Filters tweets based on privacy-related keywords
- Saves the processed tweets to the dataset

### **2. Topic Modeling (LDA)**

`LDA.py` uses **Gensim** to build an LDA model that extracts latent themes related to:

- Public data exploitation
- Personal input exploitation
- Unauthorized access

The optimal number of topics was determined to be **3**, matching the findings in the study.

### **3. Sentiment Analysis**

`Sentiment.py` computes sentiment polarity to explore how users emotionally respond to privacy concerns.

### **4. Data Categorization**

Tweets are categorized into three classes:

1. **Public Data Exploitation**
2. **Personal Input Exploitation**
3. **Unauthorized Access**

---

## ▶️ Running the Project

### **Install dependencies**

```bash
pip install numpy pandas nltk gensim textblob scikit-learn
```

Download NLTK resources (if needed):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### **Run the pipeline**

```bash
python Processing.py      # preprocessing
python Segmentation.py    # segmentation
python LDA.py             # topic modeling
python Sentiment.py       # sentiment analysis
```

---

## 📊 Dataset

The `Data/preprocessed_tweets.csv` file contains:

- Cleaned, tokenized tweets
- 11k tweets filtered for privacy discussions
- No usernames or metadata (in compliance with Twitter TOS)

This dataset was used directly in the analysis presented in the paper.

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
### **1️⃣ Data Preprocessing**
Performed using `Processing.py`, which:

- Converts text to lowercase  
- Removes links and mentions  
- Tokenizes words  
- Removes stopwords  
- Filters tweets based on privacy-related keywords  
- Saves the processed tweets to the dataset

### **2️⃣ Topic Modeling (LDA)**  
`LDA.py` uses **Gensim** to build an LDA model that extracts latent themes related to:

- Public data exploitation  
- Personal input exploitation  
- Unauthorized access  

The optimal number of topics was determined to be **3**, matching the findings in the study.

### **3️⃣ Sentiment Analysis**  
`Sentiment.py` computes sentiment polarity to explore how users emotionally respond to privacy concerns.

### **4️⃣ Data Categorization**  
Tweets are categorized into 3 classes:

1. **Public Data Exploitation**  
2. **Personal Input Exploitation**  
3. **Unauthorized Access**

---

## ▶️ Running the Project

### **Install dependencies**
```bash
pip install numpy pandas nltk gensim textblob scikit-learn
```

Download NLTK resources (if needed):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### **Run preprocessing**
```bash
python Processing.py
```

### **Run segmentation**
```bash
python Segmentation2.py
```

### **Train LDA topic model**
```bash
python LDA.py
```

### **Run sentiment analysis**
```bash
python Sentiment.py
```

---

## 📊 Dataset

The `data/preprocessed_tweets.csv` file contains:

- Cleaned, tokenized tweets  
- 11k tweets filtered for privacy discussions  
- No usernames or metadata (in compliance with Twitter TOS)

This dataset was used directly in the analysis presented in the paper.

---

## 📚 Citation

If you use this code or dataset, please cite:

**Alkamli, S., & Alabduljabbar, R. (2024).  
Understanding privacy concerns in ChatGPT: A data-driven approach with LDA topic modeling. Heliyon.**

---

## 📝 License  
This repository is provided for academic and research purposes.

