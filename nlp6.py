import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
from sklearn.ensemble import RandomForestClassifier

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt_tab')

# 1. Load and Split Data
df = pd.read_csv("spam.csv", encoding='latin1')
X = df.iloc[:, 1]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Preprocessing Function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    words = word_tokenize(text)
    list1 = []

    for word in words:
        if word not in stop_words and word.isalnum():
            list1.append(word)

    list_doc = ' '.join(list1)
    return list_doc

# Apply preprocessing
X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

# 3. TF-IDF Vectorization Logic
def dff(word, sentence):
    words = sentence.split()
    return words.count(word) / (len(words) + 1)

def idff(word, sentences):
    n = 0
    for sent in sentences:
        if word in sent:
            n += 1
    return math.log(len(sentences) / (n + 1))

all_docs = X_train.to_list() + X_test.to_list()
unique_words = set(' '.join(all_docs).split())

# Calculate IDF scores
idf_score = {}
for word in unique_words:
    idf_score[word] = idff(word, all_docs)

# Vocabulary
vocab = list(idf_score.keys())

# Transform Train Data
X_train_vector = []
for sentence in X_train:
    temp_list = []
    for w in vocab:
        tfidf_score = dff(w, sentence) * idf_score[w]
        temp_list.append(tfidf_score)
    X_train_vector.append(temp_list)

# Transform Test Data
X_test_vector = []
for sentence in X_test:
    temp_list = []
    for w in vocab:
        tfidf_score = dff(w, sentence) * idf_score[w]
        temp_list.append(tfidf_score)
    X_test_vector.append(temp_list)

# Convert to numpy arrays
X_train_vector = np.array(X_train_vector)
X_test_vector = np.array(X_test_vector)

# 4. Model Training
model = RandomForestClassifier()
model.fit(X_train_vector, y_train)

# 5. Prediction
text = input("Enter the text you want to classify: ")
text = preprocess(text)
vector = []

for word in vocab:
    tfs = dff(word, text)
    tf_idf_score = tfs * idf_score[word]
    vector.append(tf_idf_score)

output = model.predict([vector])
print(f"The classification result is: {output[0]}")
