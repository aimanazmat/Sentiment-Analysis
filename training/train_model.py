"""
Sentiment Analysis - Training Script
Phases covered: 2 (Explore) -> 8 (Save Model)
Note: Yeh script Colab mein bhi chalega. Agar Colab mein internet hai
to nltk use kar sakte hain (stemming/lemmatization ke liye), yahan
sklearn ke built-in stopwords se kaam chalaya gaya hai.
"""

import re
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# ------------------------------------------------------------
# PHASE 2: DATASET EXPLORATION
# ------------------------------------------------------------
print("=" * 60)
print("PHASE 2: DATASET EXPLORATION")
print("=" * 60)

df = pd.read_csv("../dataset/sentiment.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# PHASE 3: DATA CLEANING
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3: DATA CLEANING")
print("=" * 60)

def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)          # remove punctuation/numbers/special chars
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)
print("\nSample cleaned text:")
print(df[["text", "clean_text"]].head())


# ------------------------------------------------------------
# PHASE 4: TEXT PREPROCESSING (Tokenization + Stopword Removal)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 4: TEXT PREPROCESSING")
print("=" * 60)

# Simple tokenization (splitting by space) - TfidfVectorizer will also
# handle this internally, but we show it explicitly for learning purposes.
df["tokens"] = df["clean_text"].apply(lambda x: x.split())
print("\nSample tokens:")
print(df["tokens"].head())

# Stopword removal + Tfidf is handled together in Phase 5 using
# TfidfVectorizer(stop_words='english') for simplicity and speed.


# ------------------------------------------------------------
# PHASE 5: FEATURE ENGINEERING (TF-IDF)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 5: FEATURE ENGINEERING (TF-IDF)")
print("=" * 60)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("\nFeature matrix shape:", X.shape)
print("Sample feature names:", vectorizer.get_feature_names_out()[:10])


# ------------------------------------------------------------
# PHASE 6: MODEL TRAINING
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 6: MODEL TRAINING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained: Logistic Regression")


# ------------------------------------------------------------
# PHASE 7: MODEL EVALUATION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 7: MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted", zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("\nLabels order:", model.classes_)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ------------------------------------------------------------
# PHASE 8: SAVE THE MODEL
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 8: SAVING MODEL")
print("=" * 60)

with open("../backend/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../backend/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nSaved: backend/model.pkl")
print("Saved: backend/vectorizer.pkl")
print("\nDone! Model ready for FastAPI backend.")
