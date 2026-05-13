import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

data = pd.read_csv(r'E:\nlp\Musical_instruments_reviews 4.csv')

X = data['reviewText'].astype(str)
y = data['overall']

stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

X = X.apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

while True:

    user_review = input("\nEnter Review (or type exit): ")

    if user_review.lower() == "exit":
        print("Program Ended")
        break

    processed_review = preprocess(user_review)

    user_vector = vectorizer.transform([processed_review])

    prediction = model.predict(user_vector)[0]

    print("\nPredicted Rating:", prediction)

    if prediction <= 2:
        print("Sentiment: Bad")
    elif prediction == 3:
        print("Sentiment: Average")
    else:
        print("Sentiment: Good")