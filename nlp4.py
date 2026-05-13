import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(str(text).lower())
    return [w for w in words if w.isalnum() and w not in stop_words]

def generate_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def build_vocab(docs):
    vocab = set()
    for doc in docs:
        vocab.update(doc)
    return list(vocab)

def vectorize(docs, vocab):
    vectors = []

    for doc in docs:

        freq = {}

        for gram in doc:
            freq[gram] = freq.get(gram, 0) + 1

        vectors.append([freq.get(term, 0) for term in vocab])

    return vectors

data = pd.read_csv(
    r'C:\Users\anant\OneDrive\Documents\GitHub\3rd-year-resources-2022-scheme-rvce\6th sem\AI363IA-Natural Language processing and transformers(NLP)\LAB\Musical_instruments_reviews 4.csv'
)

x = data.iloc[:, 6]
y = data.iloc[:, 5]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1
)

n = int(input("Enter n value for n-gram: "))

x_train = [generate_ngrams(preprocess(text), n) for text in x_train]
x_test = [generate_ngrams(preprocess(text), n) for text in x_test]

vocab = build_vocab(x_train)

x_train_vec = vectorize(x_train, vocab)
x_test_vec = vectorize(x_test, vocab)

model = LogisticRegression(max_iter=1000)

model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

while True:

    review = input("\nEnter Review (or type exit): ")

    if review.lower() == "exit":
        print("Program Ended")
        break

    review_processed = generate_ngrams(preprocess(review), n)

    review_vec = vectorize([review_processed], vocab)

    pred = model.predict(review_vec)[0]

    print("\nPredicted Rating:", pred)

    if pred <= 2:
        print("Sentiment: Bad")
    elif pred == 3:
        print("Sentiment: Average")
    else:
        print("Sentiment: Good")