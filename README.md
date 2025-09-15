# Fake News Classification
This project explores the problem of classifying news articles as *fake* or *real* using Natural Language Processing (NLP). The task is approached methodically starting with exploratory analysis, preprocessing, baseline models, and progressing to stronger models like neural networks and transformers.

## Dataset
* Source: [Hugging Face](https://huggingface.co/datasets/Reyansh4/Fake-News-Classification)
* Size: 20800 rows
* Features: title, author, text
* Target: label (0:Real, 1:Fake)

## Workflow
### 1. Data Preprocessing
* Handling missing data and duplicates
* Cleaning text (quotes, contractions, punctuation, whitespace, case normalization)
* Tokenization and lemmatization (using nltk)

### 2. Exploratory Analysis & Feature Extraction
* Class distribution
* Top unigrams, bigrams, trigrams
* Unigram Word clouds for Fake vs Real
* Stylometric feature extraction and distributions
    * Lexical features: average word length, vocabulary richness
    * Syntactic features: average sentence length, punctuation ratios, POS ratios
    * Readability metrics: Flesch Readability Ease Score

### 3. Baseline Models
* TF-IDF features
* Models:
    * Logistic Regression
    * Naive Bayes
    * Linear SVC (Support Vector Classifier)
* With and without stylometric features
* Result: Linear SVC with stylometric features performed best with 96.5% accuracy.