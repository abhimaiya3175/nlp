import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv(r'Musical_instruments_reviews 4.csv')

X = data.summary.astype(str)
y = data.overall

stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

vocab = sorted(set(word for sentence in X_train for word in sentence))

word_to_index = {word: i for i, word in enumerate(vocab)}

def get_vector(words):
    vector = [0] * len(vocab)
    for word in words:
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector

train_vectors = np.array([get_vector(x) for x in X_train])
test_vectors = np.array([get_vector(x) for x in X_test])

model = LogisticRegression(max_iter=1000)
model.fit(train_vectors, y_train)

y_pred = model.predict(test_vectors)

print(classification_report(y_test, y_pred))

text = input("Enter review: ")

vector = get_vector(preprocess(text))

pred = model.predict([vector])[0]

print("Rating:", pred)

if pred <= 2:
    print("Bad")
elif pred == 3:
    print("Average")
else:
    print("Good")