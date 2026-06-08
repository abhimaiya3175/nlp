from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Musical_instruments_reviews 4.csv")
df = df[["summary", "overall"]]
df.dropna(inplace=True)

# -----------------------------
# 2. LABEL CREATION
# -----------------------------
def label_sentiment(rating):
    if rating == 5:
        return 1
    elif rating >= 3:
        return 0
    else:
        return -1

df["label"] = df["overall"].apply(label_sentiment)

# Balance dataset
min_count = df["label"].value_counts().min()
df_balanced = pd.concat([
    df[df["label"] == 1].sample(min_count, random_state=42),
    df[df["label"] == 0].sample(min_count, random_state=42),
    df[df["label"] == -1].sample(min_count, random_state=42)
])
df = df_balanced.sample(frac=1, random_state=42)

# -----------------------------
# 3. PREPROCESSING
# -----------------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'[^\w\s]', "", text.lower())
    return [word for word in text.split() if word not in stop_words]

df["tokens"] = df["summary"].apply(preprocess)

# -----------------------------
# 4. N-GRAM GENERATION
# -----------------------------
N_GRAM = 2   # change to 3 if needed

def generate_ngrams(tokens):
    ngrams = []

    # unigrams
    for token in tokens:
        ngrams.append((token,))

    # bigrams
    for i in range(len(tokens)-1):
        ngrams.append((tokens[i], tokens[i+1]))

    return ngrams

# -----------------------------
# 5. BUILD VOCABULARY
# -----------------------------
vocab = set()

for tokens in df["tokens"]:
    vocab.update(generate_ngrams(tokens))

vocab = list(vocab)
vocab_index = {gram: i for i, gram in enumerate(vocab)}

# -----------------------------
# 6. COUNT VECTORIZATION (NO TF-IDF)
# -----------------------------
def vectorize(tokens):
    vec = np.zeros(len(vocab))
    ngrams = generate_ngrams(tokens)

    for gram in ngrams:
        if gram in vocab_index:
            vec[vocab_index[gram]] += 1

    return vec

# -----------------------------
# 7. CREATE FEATURE MATRIX
# -----------------------------
X = np.array([vectorize(tokens) for tokens in df["tokens"]])
y = df["label"].values

# -----------------------------
# 8. TRAIN-TEST SPLIT
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 9. LOGISTIC REGRESSION
# -----------------------------
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def train_binary(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0

        for _ in range(self.epochs):
            z = np.dot(X, w) + b
            y_pred = self.sigmoid(z)

            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)

            w -= self.lr * dw
            b -= self.lr * db

        return w, b

    def fit(self, X, y):
        self.classes = [-1, 0, 1]
        self.models = {}

        for c in self.classes:
            y_bin = np.array([1 if label == c else 0 for label in y])
            w, b = self.train_binary(X, y_bin)
            self.models[c] = (w, b)

    def predict(self, X):
        preds = []

        for x in X:
            scores = {}
            for c in self.classes:
                w, b = self.models[c]
                scores[c] = self.sigmoid(np.dot(x, w) + b)

            preds.append(max(scores, key=scores.get))

        return preds

# -----------------------------
# 10. TRAIN MODEL
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 11. PREDICT FUNCTION
# -----------------------------
def predict_text(text):
    tokens = preprocess(text)
    vec = vectorize(tokens)
    pred = model.predict([vec])[0]

    if pred == 1:
        print("Positive")
    elif pred == 0:
        print("Neutral")
    else:
        print("Negative")
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
# -----------------------------
# 12. TEST
# -----------------------------
predict_text("The instrument is amazing")
predict_text("Average build quality and performance")
predict_text("Bad service and bad food")
predict_text("Bad bad")