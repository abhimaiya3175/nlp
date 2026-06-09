import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("Musical_instruments_reviews 4.csv")

df.dropna(subset=['reviewText'], inplace=True)

def label(x):

    if x >= 4:
        return 1            
    else:
        return 0

stop_words = set(stopwords.words('english'))

def preprocess(text):

    words = word_tokenize(str(text).lower())

    words = [
        w for w in words
        if w.isalnum() and w not in stop_words
    ]

    return " ".join(words)

X = df['reviewText'].apply(preprocess)

y = df['overall'].apply(label)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

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
    "disappointing"
}

positive_words = {
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

while True:

    review = input("\nEnter Review (or type exit): ")

    if review.lower() == "exit":

        print("\nProgram Ended")

        break

    processed_review = preprocess(review)

    words = processed_review.split()

    negative_score = 0
    positive_score = 0

    for word in words:

        if word in negative_words:

            negative_score += 1

        if word in positive_words:

            positive_score += 1

    if negative_score > positive_score:

        sentiment = "Negative"

    elif positive_score > negative_score:

        sentiment = "Positive"

    else:

        review_vec = tfidf.transform([processed_review])

        result = model.predict(review_vec)[0]

        if result == 1:

            sentiment = "Positive"

        else:

            sentiment = "Negative"

    print("\nPredicted Sentiment:", sentiment)