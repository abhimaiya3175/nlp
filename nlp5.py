import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------- LOAD DATA ---------------- #

data = pd.read_csv(r'E:\nlp\Musical_instruments_reviews 4.csv')

X = data['reviewText'].astype(str)

y = data['overall']

# ---------------- STOPWORDS ---------------- #

stop_words = set(stopwords.words('english'))

# ---------------- PREPROCESS ---------------- #

def preprocess(text):

    words = word_tokenize(text.lower())

    words = [
        word for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(words)

X = X.apply(preprocess)

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- TF-IDF FROM SCRATCH ---------------- #

def generate_ngrams(text, ngram_range=(1,2)):

    words = text.split()

    grams = []

    for n in range(ngram_range[0], ngram_range[1] + 1):

        for i in range(len(words) - n + 1):

            grams.append(" ".join(words[i:i+n]))

    return grams

# build vocabulary
vocab = {}

all_docs = []

for text in X_train:

    grams = generate_ngrams(text)

    all_docs.append(grams)

    for gram in grams:

        if gram not in vocab:

            vocab[gram] = len(vocab)

# limit features
max_features = 5000

vocab = dict(list(vocab.items())[:max_features])

# idf calculation
idf = {}

N = len(all_docs)

for word in vocab:

    count = 0

    for doc in all_docs:

        if word in doc:
            count += 1

    idf[word] = np.log((N + 1) / (count + 1)) + 1

# tf-idf vectorization
def vectorize(texts):

    vectors = []

    for text in texts:

        grams = generate_ngrams(text)

        tf = {}

        for gram in grams:

            if gram in vocab:

                tf[gram] = tf.get(gram, 0) + 1

        row = np.zeros(len(vocab))

        total = len(grams)

        if total == 0:
            vectors.append(row)
            continue

        for gram, count in tf.items():

            index = vocab[gram]

            tf_value = count / total

            row[index] = tf_value * idf[gram]

        vectors.append(row)

    return np.array(vectors)

X_train_vectors = vectorize(X_train)

X_test_vectors = vectorize(X_test)

# ---------------- MULTINOMIAL NAIVE BAYES FROM SCRATCH ---------------- #

class MultinomialNB_Scratch:

    def __init__(self, alpha=0.1):

        self.alpha = alpha

    def fit(self, X, y):

        self.classes = np.unique(y)

        n_features = X.shape[1]

        self.class_prob = {}

        self.feature_prob = {}

        total_samples = len(y)

        # calculate probabilities
        for c in self.classes:

            X_c = X[y == c]

            # prior probability
            self.class_prob[c] = len(X_c) / total_samples

            # word count
            word_count = np.sum(X_c, axis=0)

            total_words = np.sum(word_count)

            # likelihood probability
            self.feature_prob[c] = (
                word_count + self.alpha
            ) / (
                total_words + self.alpha * n_features
            )

    def predict(self, X):

        predictions = []

        for row in X:

            class_scores = {}

            for c in self.classes:

                log_prior = np.log(self.class_prob[c])

                log_likelihood = np.sum(
                    row * np.log(self.feature_prob[c] + 1e-9)
                )

                class_scores[c] = log_prior + log_likelihood

            predictions.append(
                max(class_scores, key=class_scores.get)
            )

        return np.array(predictions)

# ---------------- TRAIN MODEL ---------------- #

model = MultinomialNB_Scratch(alpha=0.1)

model.fit(X_train_vectors, y_train.values)

# ---------------- PREDICTION ---------------- #

y_pred = model.predict(X_test_vectors)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ---------------- USER INPUT ---------------- #

while True:

    user_review = input("\nEnter Review (or type exit): ")

    if user_review.lower() == "exit":

        print("Program Ended")

        break

    processed_review = preprocess(user_review)

    user_vector = vectorize([processed_review])

    prediction = model.predict(user_vector)[0]

    print("\nPredicted Rating:", prediction)

    if prediction <= 2:

        print("Sentiment: Bad")

    elif prediction == 3:

        print("Sentiment: Average")

    else:

        print("Sentiment: Good")