from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk

nltk.download('stopwords')
nltk.download('punkt')

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

data = pd.read_csv(r'C:\Users\anant\OneDrive\Documents\GitHub\3rd-year-resources-2022-scheme-rvce\6th sem\AI363IA-Natural Language processing and transformers(NLP)\LAB\Musical_instruments_reviews 4.csv')

x = data.iloc[:, 6]
y = data.iloc[:, 5]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

n = int(input("Enter n value: "))

x_train = [generate_ngrams(preprocess(text), n) for text in x_train]
x_test = [generate_ngrams(preprocess(text), n) for text in x_test]

vocab = build_vocab(x_train)

x_train_vec = vectorize(x_train, vocab)
x_test_vec = vectorize(x_test, vocab)

model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

reviews = [
    "best product. very useful",
    "Didn't fit my 1996 Fender Strat so its not that good",
    "Definitely Not For The Seasoned Piano Player what to do"
]

for review in reviews:
    review_vec = vectorize(
        [generate_ngrams(preprocess(review), n)],
        vocab
    )
    pred = model.predict(review_vec)
    print(review, "->", pred[0])