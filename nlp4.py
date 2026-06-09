import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

stop_words.discard('not')
stop_words.discard('no')

negative_words = {
    "bad","poor","worst","terrible",
    "awful","waste","broken",
    "useless","hate"
}

positive_words = {
    "excellent","amazing","great",
    "best","awesome","perfect",
    "love","fantastic"
}

def preprocess(text):

    words = word_tokenize(str(text).lower())

    words = [

        w for w in words

        if w.isalnum()
        and (
            w not in stop_words
            or w in negative_words
        )
    ]

    return " ".join(words)

print("\nLoading Dataset...\n")

data = pd.read_csv(
    "Musical_instruments_reviews 4.csv"
)

data.dropna(
    subset=['reviewText'],
    inplace=True
)

print("Dataset Loaded Successfully")

x = data['reviewText']

y = data['overall']

x = x.apply(preprocess)

x_train, x_test, y_train, y_test = train_test_split(

    x,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

n = int(
    input(
        "\nEnter N-Gram Value: "
    )
)

vectorizer = CountVectorizer(

    ngram_range=(n, n),

    max_features=8000
)

print("\nVectorizing Data...\n")

x_train_vec = vectorizer.fit_transform(
    x_train
)

x_test_vec = vectorizer.transform(
    x_test
)

print("Vectorization Completed")

model = LogisticRegression(
    max_iter=1000
)

print("\nTraining Model...\n")

model.fit(
    x_train_vec,
    y_train
)

print("Training Completed")

y_pred = model.predict(
    x_test_vec
)

print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    "\nConfusion Matrix:\n"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

def predict_review(text):

    processed = preprocess(text)

    words = processed.split()

    neg_score = 0
    pos_score = 0

    for word in words:

        if word in negative_words:

            neg_score += 2

        if word in positive_words:

            pos_score += 2

    review_vec = vectorizer.transform(
        [processed]
    )

    pred = model.predict(
        review_vec
    )[0]

    if neg_score > pos_score:

        return 1

    elif pos_score > neg_score:

        return 5

    else:

        return pred

while True:

    review = input(
        "\nEnter Review (or type exit): "
    )

    if review.lower() == "exit":

        print(
            "\nProgram Ended"
        )

        break

    pred = predict_review(
        review
    )

    print(
        "\nPredicted Rating:",
        pred
    )

    if pred <= 2:

        print(
            "Sentiment: Negative"
        )

    elif pred == 3:

        print(
            "Sentiment: Neutral"
        )

    else:

        print(
            "Sentiment: Positive"
        )