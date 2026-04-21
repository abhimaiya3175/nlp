import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# Load data
# -----------------------------
df  = pd.read_csv("../Musical_instruments_reviews 4.csv", on_bad_lines='skip')
df.dropna(subset=['reviewText'], inplace=True)

X = df['reviewText']

# -----------------------------
# FIX 1: Convert rating → sentimenta
# -----------------------------
def convert_label(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

y = df['overall'].apply(convert_label)

# -----------------------------
# Preprocessing (same as yours, optimized)
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(str(text).lower())
    return " ".join([w for w in words if w.isalnum() and w not in stop_words])

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

# -----------------------------
# TF
# -----------------------------
def tf(word, sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return words.count(word) / len(words)

# -----------------------------
# FIX 2: Correct IDF
# -----------------------------
def idf(word, sentences):
    count = sum(1 for sent in sentences if word in sent.split())
    return np.log(len(sentences) / (count + 1))

# -----------------------------
# Vocabulary (ONLY TRAIN)
# -----------------------------
all_docs = X_train.tolist()
vocab = list(set(" ".join(all_docs).split()))

idf_score = {word: idf(word, all_docs) for word in vocab}

# -----------------------------
# Vectorization
# -----------------------------
def vectorize(sentences):
    return np.array([
        [tf(word, sentence) * idf_score[word] for word in vocab]
        for sentence in sentences
    ])

X_train_vec = vectorize(X_train)
X_test_vec = vectorize(X_test)

# -----------------------------
# Model
# -----------------------------
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluation
# -----------------------------
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Prediction function
# -----------------------------
def predict_review(text):
    text = preprocess(text)
    vec = vectorize([text])
    return model.predict(vec)[0]

# -----------------------------
# Test
# -----------------------------
print(predict_review("this product is very bad and poor quality"))
print(predict_review("excellent sound quality and amazing build"))
print(predict_review("average product not great not bad"))