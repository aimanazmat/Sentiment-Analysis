# ==============================================================
# SENTIMENT ANALYSIS - Phase 2 to Phase 8
# Dataset: Twitter Entity Sentiment Analysis (Kaggle)
# Run this in Google Colab - copy each "CELL" block into a separate
# Colab cell (or run the whole file at once, it works either way)
# ==============================================================

# ---------------- CELL 1: Install & Import ----------------
!pip install wordcloud -q

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set_style("whitegrid")


# ---------------- CELL 2: Upload Dataset ----------------
# Option A: Upload directly from your laptop
from google.colab import files
uploaded = files.upload()   # select twitter_training.csv and twitter_validation.csv here

# Option B (alternative): if dataset is in Google Drive, use this instead:
# from google.colab import drive
# drive.mount('/content/drive')
# train_path = "/content/drive/MyDrive/archive (1)/twitter_training.csv"
# val_path   = "/content/drive/MyDrive/archive (1)/twitter_validation.csv"


# ==============================================================
# PHASE 2: DATASET EXPLORATION
# ==============================================================

# This dataset has NO header row, so we name the columns manually
columns = ["tweet_id", "entity", "sentiment", "text"]

df_train = pd.read_csv("twitter_training.csv", names=columns)
df_val = pd.read_csv("twitter_validation.csv", names=columns)

print("Training set shape:", df_train.shape)
print("Validation set shape:", df_val.shape)

print("\nFirst 5 rows:")
print(df_train.head())

print("\nDataset info:")
print(df_train.info())

print("\nMissing values:")
print(df_train.isnull().sum())

# We only need text + sentiment for this project
df = df_train[["text", "sentiment"]].copy()

# Merge "Irrelevant" into "Neutral" (dataset creators treat them the same way)
df["sentiment"] = df["sentiment"].replace("Irrelevant", "Neutral")

# Drop missing text rows
df = df.dropna(subset=["text"])

print("\nFinal label distribution:")
print(df["sentiment"].value_counts())

# 📊 VISUALIZATION 1: Label Distribution (Bar Chart)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sentiment", order=df["sentiment"].value_counts().index, palette="viridis")
plt.title("Sentiment Label Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.show()

# 📊 VISUALIZATION 2: Label Distribution (Pie Chart)
plt.figure(figsize=(6, 6))
df["sentiment"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("viridis"))
plt.title("Sentiment Proportion")
plt.ylabel("")
plt.tight_layout()
plt.savefig("label_pie_chart.png")
plt.show()

# 📊 VISUALIZATION 3: Tweet length distribution
df["text_length"] = df["text"].astype(str).apply(len)
plt.figure(figsize=(7, 4))
sns.histplot(df["text_length"], bins=40, kde=True, color="teal")
plt.title("Tweet Length Distribution")
plt.xlabel("Character Length")
plt.tight_layout()
plt.savefig("tweet_length_distribution.png")
plt.show()


# ==============================================================
# PHASE 3: DATA CLEANING
# ==============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove mentions
    text = re.sub(r"[^a-z\s]", "", text)               # remove punctuation/numbers/special chars
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("\nSample cleaned text:")
print(df[["text", "clean_text"]].head())

# Remove rows that became empty after cleaning
df = df[df["clean_text"].str.len() > 0]


# ==============================================================
# PHASE 4: TEXT PREPROCESSING (Tokenization + Stopwords)
# ==============================================================

df["tokens"] = df["clean_text"].apply(lambda x: x.split())
print("\nSample tokens:")
print(df["tokens"].head())

# 📊 VISUALIZATION 4: WordCloud of most frequent words
all_words = " ".join(df["clean_text"].sample(min(5000, len(df)), random_state=42))
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Tweets")
plt.tight_layout()
plt.savefig("wordcloud.png")
plt.show()

# Stopword removal is handled inside TfidfVectorizer(stop_words='english') in Phase 5


# ==============================================================
# PHASE 5: FEATURE ENGINEERING (TF-IDF)
# ==============================================================

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

print("\nFeature matrix shape:", X.shape)
print("Sample feature names:", vectorizer.get_feature_names_out()[:15])


# ==============================================================
# PHASE 6: MODEL TRAINING
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained: Logistic Regression")


# ==============================================================
# PHASE 7: MODEL EVALUATION
# ==============================================================

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nAccuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 📊 VISUALIZATION 5: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 📊 VISUALIZATION 6: Evaluation Metrics Bar Chart
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [acc, prec, rec, f1]
})
plt.figure(figsize=(6, 4))
sns.barplot(data=metrics_df, x="Metric", y="Score", palette="mako")
plt.ylim(0, 1)
plt.title("Model Evaluation Metrics")
for i, v in enumerate(metrics_df["Score"]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("evaluation_metrics.png")
plt.show()


# ==============================================================
# PHASE 8: SAVE THE MODEL
# ==============================================================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nSaved: model.pkl")
print("Saved: vectorizer.pkl")
print("\nDone! Download these two files and put them in your backend/ folder.")

# Download the model files directly from Colab (optional)
from google.colab import files
files.download("model.pkl")
files.download("vectorizer.pkl")
