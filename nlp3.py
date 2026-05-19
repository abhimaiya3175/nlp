import numpy as np
import pandas as pd
import nltk
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------- DOWNLOAD NLTK DATA ---------------- #

nltk.download('punkt')
nltk.download('stopwords')

# ---------------- START TIMER ---------------- #

start_time = time.time()

# ---------------- LOAD DATASET ---------------- #

print("\nLoading Dataset...\n")

df = pd.read_csv(
    r"E:\nlp\Musical_instruments_reviews 4.csv",
    on_bad_lines='skip'
)

# remove null values
df.dropna(subset=['reviewText'], inplace=True)

print("Dataset Loaded Successfully")

# ---------------- BALANCED DATASET ---------------- #

print("\nCreating Balanced Dataset...\n")

# positive reviews
positive_df = df[df['overall'] >= 4]

# negative reviews
negative_df = df[df['overall'] <= 2]

# take equal sample size
sample_size = min(
    len(positive_df),
    len(negative_df)
)

positive_df = positive_df.sample(
    sample_size,
    random_state=42
)

negative_df = negative_df.sample(
    sample_size,
    random_state=42
)

# combine balanced data
df = pd.concat([
    positive_df,
    negative_df
])

# shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
)

print("Balanced Dataset Created")

print(
    "\nPositive Reviews:",
    len(positive_df)
)

print(
    "Negative Reviews:",
    len(negative_df)
)

# ---------------- LABEL CONVERSION ---------------- #

# positive = 1
# negative = 0

def convert_label(r):

    if r >= 4:

        return 1

    else:

        return 0

# ---------------- INPUT AND OUTPUT ---------------- #

X = df['reviewText']

y = df['overall'].apply(convert_label)

# ---------------- PREPROCESSING ---------------- #

stop_words = set(stopwords.words('english'))

# keep negation words
stop_words.discard('not')
stop_words.discard('no')

# important negative words
negative_words = {
    "bad",
    "poor",
    "worst",
    "waste",
    "terrible",
    "awful",
    "useless",
    "broken",
    "hate",
    "disappointing",
    "boring"
}

def preprocess(text):

    words = word_tokenize(
        str(text).lower()
    )

    cleaned_words = []

    for word in words:

        if (
            word.isalnum()
            and (
                word not in stop_words
                or word in negative_words
            )
        ):

            cleaned_words.append(word)

    return " ".join(cleaned_words)

print("\nPreprocessing Data...\n")

# preprocess all reviews
X = X.apply(preprocess)

# ---------------- TRAIN TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------- TF FUNCTION ---------------- #

def tf(word, sentence):

    words = sentence.split()

    if len(words) == 0:

        return 0

    return words.count(word) / len(words)

# ---------------- IDF FUNCTION ---------------- #

def idf(word, sentences):

    count = 0

    for sentence in sentences:

        if word in sentence.split():

            count += 1

    return np.log(
        len(sentences) / (count + 1)
    )

# ---------------- BUILD VOCAB ---------------- #

print("\nBuilding Vocabulary...\n")

all_docs = X_train.tolist()

vocab = list(
    set(
        " ".join(all_docs).split()
    )
)

# larger vocabulary
vocab = vocab[:10000]

print(
    "Vocabulary Size:",
    len(vocab)
)

# ---------------- IDF SCORES ---------------- #

idf_score = {}

for word in vocab:

    idf_score[word] = idf(
        word,
        all_docs
    )

# ---------------- TF-IDF VECTORIZE ---------------- #

def vectorize(sentences):

    vectors = []

    for sentence in sentences:

        row = []

        for word in vocab:

            value = (
                tf(word, sentence)
                * idf_score[word]
            )

            row.append(value)

        vectors.append(row)

    return np.array(vectors)

print("\nVectorizing Data...\n")

X_train_vec = vectorize(X_train)

X_test_vec = vectorize(X_test)

print("Vectorization Completed")

# ---------------- LOGISTIC REGRESSION ---------------- #

class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.1,
        epochs=300
    ):

        self.learning_rate = learning_rate

        self.epochs = epochs

    # ---------------- SIGMOID FUNCTION ---------------- #

    def sigmoid(self, z):

        return 1 / (
            1 + np.exp(-z)
        )

    # ---------------- TRAIN MODEL ---------------- #

    def fit(self, X, y):

        rows, cols = X.shape

        # initialize weights
        self.W = np.zeros(cols)

        # initialize bias
        self.b = 0

        print("\nTraining Started...\n")

        # gradient descent
        for epoch in range(self.epochs):

            # linear equation
            z = np.dot(X, self.W) + self.b

            # prediction probability
            y_hat = self.sigmoid(z)

            # gradients
            dW = (
                1 / rows
            ) * np.dot(
                X.T,
                (y_hat - y)
            )

            db = (
                1 / rows
            ) * np.sum(
                y_hat - y
            )

            # update weights
            self.W = (
                self.W
                - self.learning_rate * dW
            )

            # update bias
            self.b = (
                self.b
                - self.learning_rate * db
            )

            # print loss
            if epoch % 20 == 0:

                loss = -np.mean(
                    y * np.log(y_hat + 1e-9)
                    +
                    (1 - y)
                    * np.log(
                        1 - y_hat + 1e-9
                    )
                )

                print(
                    f"Epoch {epoch} Loss = {loss}"
                )

        print("\nTraining Completed...\n")

    # ---------------- PREDICT ---------------- #

    def predict(self, X):

        z = np.dot(X, self.W) + self.b

        y_hat = self.sigmoid(z)

        predictions = []

        for prob in y_hat:

            if prob >= 0.5:

                predictions.append(1)

            else:

                predictions.append(0)

        return np.array(predictions)

# ---------------- CREATE MODEL ---------------- #

model = LogisticRegressionScratch(
    learning_rate=0.1,
    epochs=300
)

# ---------------- TRAIN MODEL ---------------- #

model.fit(
    X_train_vec,
    y_train.values
)

# ---------------- TEST MODEL ---------------- #

print("\nTesting Model...\n")

y_pred = model.predict(X_test_vec)

# accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    accuracy
)

# classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# confusion matrix
print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ---------------- REVIEW PREDICTION ---------------- #

def predict_review(text):

    # strong negative keywords
    negative_keywords = {
        "bad",
        "poor",
        "worst",
        "waste",
        "terrible",
        "awful",
        "useless",
        "broken",
        "hate",
        "disappointing"
    }

    # strong positive keywords
    positive_keywords = {
        "excellent",
        "amazing",
        "great",
        "best",
        "good",
        "awesome",
        "perfect",
        "love",
        "fantastic"
    }

    # preprocess review
    processed_text = preprocess(text)

    words = processed_text.split()

    # keyword scores
    negative_score = 0

    positive_score = 0

    for word in words:

        if word in negative_keywords:

            negative_score += 2

        if word in positive_keywords:

            positive_score += 2

    # ML prediction
    vec = vectorize([processed_text])

    pred = model.predict(vec)[0]

    # combine rule-based + ML
    if negative_score > positive_score:

        return "negative"

    elif positive_score > negative_score:

        return "positive"

    else:

        if pred == 1:

            return "positive"

        else:

            return "negative"

# ---------------- USER INPUT ---------------- #

while True:

    review = input(
        "\nEnter Review (or type exit): "
    )

    if review.lower() == "exit":

        print("\nProgram Ended")

        break

    result = predict_review(review)

    print(
        "\nPredicted Sentiment:",
        result
    )

# ---------------- EXECUTION TIME ---------------- #

end_time = time.time()

print(
    "\nTotal Execution Time:",
    round(
        end_time - start_time,
        2
    ),
    "seconds"
)