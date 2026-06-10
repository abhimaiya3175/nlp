import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv(
    "Musical_instruments_reviews 4.csv"
)

stop_words = set(
    stopwords.words('english')
)

def preprocess(text):

    words = word_tokenize(
        str(text).lower()
    )

    words = [
        w for w in words
        if w.isalpha()
        and w not in stop_words
    ]

    return " ".join(words)

X = data['reviewText'].apply(
    preprocess
)

y = data['overall']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)

X_train = tfidf.fit_transform(
    X_train
)

X_test = tfidf.transform(
    X_test
)

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

while True:

    review = input(
        "\nEnter Review: "
    )

    if review.lower() == "exit":

        break

    review = preprocess(
        review
    )

    review = tfidf.transform(
        [review]
    )

    pred = model.predict(
        review
    )[0]

    print(
        "Predicted Rating:",
        pred
    )