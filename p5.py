# SENTIMENT ANALYSIS USING CUSTOM BAG OF WORDS + CUSTOM NAIVE BAYES
# No inbuilt Bag of Words or Naive Bayes libraries used

import pandas as pd
import re
import math
import random
from collections import defaultdict

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

data = pd.read_csv("Musical_instruments_reviews 4.csv")

data = data[["summary", "overall"]]
data.dropna(inplace=True)

# ------------------------------------------------------------
# 2. LABEL CONVERSION
# 1,2 -> Negative
# 3   -> Neutral
# 4,5 -> Positive
# ------------------------------------------------------------

def label_sentiment(rating):
    if rating == 5:
        return "Positive"
    elif rating >= 3:
        return "Neutral"
    else:
        return "Negative"

data["label"] = data["overall"].apply(label_sentiment)

min_count = data["label"].value_counts().min()
data_balanced = pd.concat([
    data[data["label"] == "Positive"].sample(min_count, random_state=42),
    data[data["label"] == "Neutral"].sample(min_count, random_state=42),
    data[data["label"] == "Negative"].sample(min_count, random_state=42)
])
data = data_balanced.sample(frac=1, random_state=42)


# ------------------------------------------------------------
# 3. TEXT PREPROCESSING (NO NLTK)
# ------------------------------------------------------------

def preprocess(text):
    text = str(text).lower()
    
    # remove punctuation/numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # tokenize
    words = text.split()
    
    return words

# ------------------------------------------------------------
# 4. TRAIN TEST SPLIT (MANUAL)
# ------------------------------------------------------------

dataset = list(zip(data["summary"], data["label"]))

random.seed(42)
random.shuffle(dataset)

split_index = int(0.8 * len(dataset))

train_data = dataset[:split_index]
test_data = dataset[split_index:]

print("Training samples:", len(train_data))
print("Testing samples :", len(test_data))

# ------------------------------------------------------------
# 5. BUILD VOCABULARY (CUSTOM BAG OF WORDS)
# ------------------------------------------------------------

vocab = set()

for text, label in train_data:
    words = preprocess(text)
    for word in words:
        vocab.add(word)

vocab = sorted(list(vocab))

# word -> index mapping
vocab_index = {}

for i, word in enumerate(vocab):
    vocab_index[word] = i

print("Vocabulary size:", len(vocab))

# ------------------------------------------------------------
# 6. CUSTOM BAG OF WORDS VECTOR CREATION
# ------------------------------------------------------------

def text_to_bow(text, vocab_index):
    words = preprocess(text)
    
    vector = [0] * len(vocab_index)

    for word in words:
        if word in vocab_index:
            vector[vocab_index[word]] += 1

    return vector

# ------------------------------------------------------------
# 7. CUSTOM NAIVE BAYES CLASSIFIER
# ------------------------------------------------------------

class CustomNaiveBayes:

    def __init__(self):
        self.class_priors = {}
        self.word_counts = {}
        self.total_words_per_class = {}
        self.classes = set()
        self.vocab_size = 0

    def fit(self, train_data, vocab):

        self.vocab_size = len(vocab)

        class_doc_count = defaultdict(int)

        for text, label in train_data:

            self.classes.add(label)
            class_doc_count[label] += 1

            if label not in self.word_counts:
                self.word_counts[label] = defaultdict(int)
                self.total_words_per_class[label] = 0

            words = preprocess(text)

            for word in words:
                self.word_counts[label][word] += 1
                self.total_words_per_class[label] += 1

        total_docs = len(train_data)

        # Prior probabilities
        for label in self.classes:
            self.class_priors[label] = class_doc_count[label] / total_docs

    def predict(self, text):

        words = preprocess(text)

        scores = {}

        for label in self.classes:

            # log prior
            score = math.log(self.class_priors[label])

            total_words = self.total_words_per_class[label]

            for word in words:

                word_freq = self.word_counts[label].get(word, 0)

                # Laplace smoothing
                probability = (word_freq + 1) / (total_words + self.vocab_size)

                score += math.log(probability)

            scores[label] = score

        prediction = max(scores, key=scores.get)

        return prediction

# ------------------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------------------

model = CustomNaiveBayes()
model.fit(train_data, vocab)

print("\nModel training completed.")

# ------------------------------------------------------------
# 9. TESTING
# ------------------------------------------------------------

test_sentences = [
    "This product is amazing and works perfectly",
    "Terrible quality and very disappointing",
    "The product is okay and average"
]

# ------------------------------------------------------------
# 10. EVALUATION (MANUAL ACCURACY)
# ------------------------------------------------------------

correct = 0
total = len(test_data)

for text, actual_label in test_data:
    predicted_label = model.predict(text)

    if predicted_label == actual_label:
        correct += 1

accuracy = (correct / total) * 100
print("Model Accuracy:", round(accuracy, 2), "%")

# ------------------------------------------------------------
# 11. USER INPUT PREDICTION
# ------------------------------------------------------------

while True:
    user_text = input("\nEnter review text (or type 'exit'): ")

    if user_text.lower() == "exit":
        break

    prediction = model.predict(user_text)
    print("Predicted Sentiment:", prediction)