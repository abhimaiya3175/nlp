import pandas as pd
import numpy as np
import math
import re
from nltk.corpus import stopwords

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Musical_instruments_reviews 4.csv")

df = df[['summary', 'overall']]
df.dropna(inplace=True)

# Convert rating → sentiment
def label_sentiment(rating):
    if rating >= 4:
        return 1   # Positive
    elif rating == 3:
        return 0   # Neutral
    else:
        return -1  # Negative

df['label'] = df['overall'].apply(label_sentiment)
# -----------------------------
# BALANCE DATASET
# -----------------------------
min_count = df['label'].value_counts().min()

df_balanced = pd.concat([
    df[df['label'] == 1].sample(min_count, random_state=42),
    df[df['label'] == 0].sample(min_count, random_state=42),
    df[df['label'] == -1].sample(min_count, random_state=42)
])

df = df_balanced.sample(frac=1, random_state=42)  # shuffle
# -----------------------------
# 2. PREPROCESS TEXT
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words

df['tokens'] = df['summary'].apply(preprocess)

# -----------------------------
# 3. BUILD VOCAB
# -----------------------------
vocab = set()
for tokens in df['tokens']:
    vocab.update(tokens)

vocab = list(vocab)
vocab_index = {word: i for i, word in enumerate(vocab)}

# -----------------------------
# 4. TF CALCULATION
# -----------------------------
def compute_tf(tokens):
    tf = np.zeros(len(vocab))
    word_count = len(tokens)

    for word in tokens:
        if word in vocab_index:
            tf[vocab_index[word]] += 1

    if word_count > 0:
        tf = tf / word_count

    return tf

# -----------------------------
# 5. IDF CALCULATION
# -----------------------------
N = len(df)
idf = np.zeros(len(vocab))

for word, i in vocab_index.items():
    doc_count = 0
    for tokens in df['tokens']:
        if word in tokens:
            doc_count += 1
    idf[i] = math.log((N + 1) / (doc_count + 1))

# -----------------------------
# 6. TF-IDF MATRIX
# -----------------------------
X = []

for tokens in df['tokens']:
    tf = compute_tf(tokens)
    tfidf = tf * idf
    X.append(tfidf)

X = np.array(X)
y = df['label'].values

# -----------------------------
# 7. TRAIN-TEST SPLIT
# -----------------------------
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class LogisticRegressionOVR:
    def __init__(self, lr=0.5, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train_binary(self, X, y):
        weights = np.zeros(X.shape[1])
        bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, weights) + bias
            y_pred = self.sigmoid(linear)

            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)

            weights -= self.lr * dw
            bias -= self.lr * db

        return weights, bias

    def fit(self, X, y):
        self.classes = [-1, 0, 1]
        self.models = {}

        for c in self.classes:
            y_bin = np.array([1 if i == c else 0 for i in y])
            w, b = self.train_binary(X, y_bin)
            self.models[c] = (w, b)

    def predict(self, X):
        results = []

        for x in X:
            scores = {}
            for c in self.classes:
                w, b = self.models[c]
                score = self.sigmoid(np.dot(x, w) + b)
                scores[c] = score

            results.append(max(scores, key=scores.get))

        return results

model = LogisticRegressionOVR()
model.fit(X_train, y_train)

preds = model.predict(X_test)

# Convert back to -1 / 0 / 1 approx
y_test_bin = [1 if i == 1 else 0 for i in y_test]

accuracy = sum([1 for i in range(len(preds)) if preds[i] == y_test[i]]) / len(preds)
print("Accuracy:", accuracy)

# -----------------------------
# 10. INFERENCE FUNCTION
# -----------------------------
def predict_text(text):
    tokens = preprocess(text)
    tf = compute_tf(tokens)
    tfidf = tf * idf

    pred = model.predict([tfidf])[0]

    if pred == 1:
        return "Positive"
    elif pred == 0:
        return "Neutral"
    else:
        return "Negative"
        
# -----------------------------
# 11. TEST WITH GIVEN EXAMPLES
# -----------------------------
print(predict_text("The food was amazing and staff was friendly"))
print(predict_text("Terrible service and bad food"))
print(predict_text("Food was okay, nothing special"))
print(predict_text("Food was bad"))