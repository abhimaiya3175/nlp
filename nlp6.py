import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv(
    "spam.csv",
    encoding="latin1"
)

X = data.iloc[:,1]

y = data.iloc[:,0]

stop_words = set(
    stopwords.words('english')
)

def preprocess(text):

    words = word_tokenize(
        str(text).lower()
    )

    words = [
        w for w in words
        if w.isalnum()
        and w not in stop_words
    ]

    return " ".join(words)

X = X.apply(preprocess)

le = LabelEncoder()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

tfidf = TfidfVectorizer()

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100
)

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

text = input(
    "Enter Message: "
)

text = preprocess(text)

text = tfidf.transform(
    [text]
)

pred = model.predict(
    text
)

print(
    "Result:",
    le.inverse_transform(pred)[0]
)