import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Added for custom Random Forest
from sklearn.preprocessing import LabelEncoder # Added for label encoding

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math

# Removed: from sklearn.ensemble import RandomForestClassifier as it is a built-in function

df = pd.read_csv("spam.csv", encoding='latin1')
X=df.iloc[:,1]
y=df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform y_train and transform y_test
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Update y_train and y_test to their encoded versions
y_train = y_train_encoded
y_test = y_test_encoded

print("Labels 'ham' and 'spam' have been encoded to numerical values.")
print(f"Encoded y_train labels (first 5): {y_train[:5]}")

df

def preprocess(text):
    stop_words=set(stopwords.words('english'))
    text=text.lower()
    words=word_tokenize(text)
    list1=[]

    for word in words:
        if word not in stop_words and word.isalnum():
            list1.append(word)

    list_doc=' '.join(list1)
    return list_doc

X_train=X_train.apply(preprocess)
X_test=X_test.apply(preprocess)

def dff(word,sentence):
    words=sentence.split()
    return words.count(word)/(len(words)+1)

def idff(word,sentences):
    n=0
    for sent in sentences:
        if word in sent:
            n+=1
    return math.log(len(sentences)/(n+1))

all_docs=X_train.to_list()+X_test.to_list()

unique_words=set(' '.join(all_docs).split())

idf_score={}
for word in unique_words:
    idf_score[word]=idff(word,all_docs)

#vocabulary
vocab=list(idf_score.keys())

X_train_vector=[]
for sentence in X_train:
    temp_list=[]
    for w in vocab:
        tfidf_score=dff(w,sentence)*idf_score[w]
        temp_list.append(tfidf_score)
    X_train_vector.append(temp_list)


X_test_vector = []
for sentence in X_test:
    temp_list = []
    for w in vocab:
        tfidf_score = dff(w, sentence)*idf_score[w]
        temp_list.append(tfidf_score)
    X_test_vector.append(temp_list)

X_train_vector=np.array(X_train_vector)
X_test_vector=np.array(X_test_vector)

from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()


y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)


y_train = y_train_encoded
y_test = y_test_encoded

print("Labels 'ham' and 'spam' have been encoded to numerical values.")
print(f"Encoded y_train labels (first 5): {y_train[:5]}")
print(f"Original labels: {le.inverse_transform(np.unique(y_train))}")

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class SimpleRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, max_depth=10, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.classes_ = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.trees])


        final_predictions = []
        for i in range(X.shape[0]):

            sample_predictions = predictions[:, i]

            counts = {cls: 0 for cls in self.classes_}
            for pred in sample_predictions:
                if pred in counts:
                    counts[pred] += 1
            final_predictions.append(max(counts, key=counts.get))
        return np.array(final_predictions)


model = SimpleRandomForest(n_estimators=50, max_depth=10, random_state=42) 
model.fit(X_train_vector, y_train)

print(f"Custom SimpleRandomForest trained with {model.n_estimators} trees.")

from sklearn.metrics import accuracy_score


y_pred = model.predict(X_test_vector)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")


le = LabelEncoder()
le.fit(y)

print(f"LabelEncoder re-fitted with classes: {le.classes_}")

text = input("Enter the text you want to classify: ")

processed_text = preprocess(text)
# print(f"Preprocessed input text: {processed_text}")
# print(f"Sample of vocabulary (first 50 words): {vocab[:50]}")

vector=[]

for word in vocab:
    tfs=dff(word,processed_text)
    tf_idf_score=tfs*idf_score[word]
    vector.append(tf_idf_score)

# print(f"Generated feature vector (first 10 elements): {vector[:10]}")

output_numeric = model.predict([vector])
print(f"Raw numerical prediction from model: {output_numeric[0]}")

output_label = le.inverse_transform(output_numeric)
print(f"The classification result is: {output_label[0]}")