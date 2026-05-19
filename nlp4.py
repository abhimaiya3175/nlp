import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------- DOWNLOAD NLTK DATA ---------------- #

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------- STOPWORDS ---------------- #

stop_words = set(stopwords.words('english'))

# keep important negation words
stop_words.discard('not')
stop_words.discard('no')

# important negative words
negative_words = {
    "bad",
    "poor",
    "worst",
    "terrible",
    "awful",
    "waste",
    "broken",
    "useless",
    "hate"
}

# ---------------- PREPROCESS FUNCTION ---------------- #

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

    return cleaned_words

# ---------------- NGRAM GENERATION ---------------- #

def generate_ngrams(words, n):

    grams = []

    for i in range(
        len(words) - n + 1
    ):

        grams.append(
            tuple(words[i:i+n])
        )

    return grams

# ---------------- BUILD VOCAB ---------------- #

def build_vocab(docs):

    vocab = set()

    for doc in docs:

        vocab.update(doc)

    return list(vocab)

# ---------------- VECTORIZE FUNCTION ---------------- #

def vectorize(docs, vocab):

    vectors = []

    for doc in docs:

        freq = {}

        # count frequency
        for gram in doc:

            freq[gram] = (
                freq.get(gram, 0) + 1
            )

        row = []

        for term in vocab:

            row.append(
                freq.get(term, 0)
            )

        vectors.append(row)

    return np.array(vectors)

# ---------------- LOAD DATA ---------------- #

print("\nLoading Dataset...\n")

data = pd.read_csv(
    r'E:\nlp\Musical_instruments_reviews 4.csv'
)

print("Dataset Loaded Successfully")

# ---------------- REMOVE NULL VALUES ---------------- #

data.dropna(
    subset=['reviewText'],
    inplace=True
)

# ---------------- BALANCED DATASET ---------------- #

print("\nCreating Balanced Dataset...\n")

rating_1 = data[data.iloc[:, 5] == 1.0]

rating_2 = data[data.iloc[:, 5] == 2.0]

rating_3 = data[data.iloc[:, 5] == 3.0]

rating_4 = data[data.iloc[:, 5] == 4.0]

rating_5 = data[data.iloc[:, 5] == 5.0]

# minimum class size
sample_size = min(
    len(rating_1),
    len(rating_2),
    len(rating_3),
    len(rating_4),
    len(rating_5)
)

# equal sampling
rating_1 = rating_1.sample(
    sample_size,
    random_state=42
)

rating_2 = rating_2.sample(
    sample_size,
    random_state=42
)

rating_3 = rating_3.sample(
    sample_size,
    random_state=42
)

rating_4 = rating_4.sample(
    sample_size,
    random_state=42
)

rating_5 = rating_5.sample(
    sample_size,
    random_state=42
)

# combine all classes
data = pd.concat([
    rating_1,
    rating_2,
    rating_3,
    rating_4,
    rating_5
])

# shuffle dataset
data = data.sample(
    frac=1,
    random_state=42
)

print("Balanced Dataset Created")

print("\nClass Distribution:\n")

print(
    data.iloc[:, 5].value_counts()
)

# ---------------- INPUT AND OUTPUT ---------------- #

# review text
x = data.iloc[:, 6]

# ratings
y = data.iloc[:, 5]

# ---------------- TRAIN TEST SPLIT ---------------- #

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

# ---------------- USER INPUT FOR NGRAM ---------------- #

n = int(
    input(
        "\nEnter n value for n-gram: "
    )
)

# ---------------- PREPROCESS DATA ---------------- #

print("\nPreprocessing Data...\n")

x_train = [

    generate_ngrams(
        preprocess(text),
        n
    )

    for text in x_train
]

x_test = [

    generate_ngrams(
        preprocess(text),
        n
    )

    for text in x_test
]

# ---------------- BUILD VOCAB ---------------- #

print("\nBuilding Vocabulary...\n")

vocab = build_vocab(x_train)

# increase vocabulary size
vocab = vocab[:8000]

print(
    "Vocabulary Size:",
    len(vocab)
)

# ---------------- VECTORIZE ---------------- #

print("\nVectorizing Data...\n")

x_train_vec = vectorize(
    x_train,
    vocab
)

x_test_vec = vectorize(
    x_test,
    vocab
)

print("Vectorization Completed")

# ---------------- LOGISTIC REGRESSION ---------------- #

class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=300
    ):

        self.learning_rate = learning_rate

        self.epochs = epochs

    # ---------------- SOFTMAX FUNCTION ---------------- #

    def softmax(self, z):

        exp_z = np.exp(
            z - np.max(
                z,
                axis=1,
                keepdims=True
            )
        )

        return exp_z / np.sum(
            exp_z,
            axis=1,
            keepdims=True
        )

    # ---------------- ONE HOT ENCODING ---------------- #

    def one_hot(self, y, classes):

        y_encoded = np.zeros(
            (len(y), classes)
        )

        for i in range(len(y)):

            label = int(y.iloc[i]) - 1

            y_encoded[i][label] = 1

        return y_encoded

    # ---------------- TRAIN MODEL ---------------- #

    def fit(self, X, y):

        rows, cols = X.shape

        self.classes = len(
            np.unique(y)
        )

        # initialize weights
        self.W = np.zeros(
            (cols, self.classes)
        )

        # initialize bias
        self.b = np.zeros(
            (1, self.classes)
        )

        # one hot encoding
        y_encoded = self.one_hot(
            y,
            self.classes
        )

        print("\nTraining Started...\n")

        # gradient descent
        for epoch in range(self.epochs):

            # linear equation
            z = np.dot(
                X,
                self.W
            ) + self.b

            # prediction probability
            y_hat = self.softmax(z)

            # gradients
            dW = (
                1 / rows
            ) * np.dot(
                X.T,
                (y_hat - y_encoded)
            )

            db = (
                1 / rows
            ) * np.sum(
                (y_hat - y_encoded),
                axis=0,
                keepdims=True
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
                    np.sum(
                        y_encoded
                        * np.log(
                            y_hat + 1e-9
                        ),
                        axis=1
                    )
                )

                print(
                    f"Epoch {epoch} Loss = {loss}"
                )

        print("\nTraining Completed...\n")

    # ---------------- PREDICT FUNCTION ---------------- #

    def predict(self, X):

        z = np.dot(
            X,
            self.W
        ) + self.b

        y_hat = self.softmax(z)

        predictions = np.argmax(
            y_hat,
            axis=1
        )

        return predictions + 1

# ---------------- CREATE MODEL ---------------- #

model = LogisticRegressionScratch(
    learning_rate=0.01,
    epochs=300
)

# ---------------- TRAIN MODEL ---------------- #

model.fit(
    x_train_vec,
    y_train
)

# ---------------- TEST MODEL ---------------- #

print("\nTesting Model...\n")

y_pred = model.predict(
    x_test_vec
)

# ---------------- ACCURACY ---------------- #

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    accuracy
)

# ---------------- CLASSIFICATION REPORT ---------------- #

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ---------------- CONFUSION MATRIX ---------------- #

print(
    "\nConfusion Matrix:\n"
)

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
        "terrible",
        "awful",
        "waste",
        "broken",
        "useless",
        "hate"
    }

    # strong positive keywords
    positive_keywords = {
        "excellent",
        "amazing",
        "great",
        "best",
        "awesome",
        "perfect",
        "love",
        "fantastic"
    }

    # preprocess review
    processed_review = preprocess(text)

    # keyword scores
    negative_score = 0

    positive_score = 0

    for word in processed_review:

        if word in negative_keywords:

            negative_score += 2

        if word in positive_keywords:

            positive_score += 2

    # generate ngrams
    processed_review = generate_ngrams(
        processed_review,
        n
    )

    # vectorize review
    review_vec = vectorize(
        [processed_review],
        vocab
    )

    # ML prediction
    pred = model.predict(
        review_vec
    )[0]

    # combine ML + rule based
    if negative_score > positive_score:

        pred = 1.0

    elif positive_score > negative_score:

        pred = 5.0

    return pred

# ---------------- USER REVIEW INPUT ---------------- #

while True:

    review = input(
        "\nEnter Review (or type exit): "
    )

    if review.lower() == "exit":

        print("\nProgram Ended")

        break

    pred = predict_review(review)

    print(
        "\nPredicted Rating:",
        pred
    )

    # sentiment
    if pred <= 2:

        print("Sentiment: Negative")

    elif pred == 3:

        print("Sentiment: Neutral")

    else:

        print("Sentiment: Positive")