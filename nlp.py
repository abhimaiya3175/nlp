import nltk
from nltk.corpus import stopwords
import numpy as np
import re

nltk.download('stopwords')


# -------------------------------
# Custom sentence tokenizer
# -------------------------------
def sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


# -------------------------------
# Custom word tokenizer
# -------------------------------
def word_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.split()


def tf_idf(sentence, sentences):
    
    # preprocess the text
    def preprocess(text):
        words = word_tokenize(text)
        s_w = set(stopwords.words("english"))
        
        cleaned = []
        for word in words:
            if word not in s_w:
                if word.isalnum():
                    cleaned.append(word)
        return cleaned

    # calculate tf in sentence
    def tf(word, sentence):
        words = preprocess(sentence)
        return words.count(word) / (len(words) if len(words) > 0 else 1)

    # calculate idf in the whole document
    def idf(word, sentences):
        count = 0

        for sent in sentences:
            if word in preprocess(sent):
                count += 1

        return np.log(len(sentences) / (count + 1))

    words = preprocess(sentence)

    tf_idf_score = 0

    for word in words:
        tf_score = tf(word, sentence)
        idf_score = idf(word, sentences)
        tf_idf_score += tf_score * idf_score

    return tf_idf_score / (len(words) if len(words) > 0 else 1)


def summarizer(text, length):
    
    sent_score = {}

    sentences = sent_tokenize(text)

    for sentence in sentences:
        score = tf_idf(sentence, sentences)
        sent_score[sentence] = score

    # sort sentences based on score
    sorted_sent_score = dict(sorted(sent_score.items(), key=lambda x: x[1], reverse=True))

    top_k = list(sorted_sent_score.keys())[:length]

    # preserve original order
    top_k = sorted(top_k, key=lambda x: sentences.index(x))

    summary = " ".join(top_k)

    return summary


# Example
text = """
Born in Ranchi, Dhoni made his first class debut for Bihar in 1999.
He made his debut for the Indian cricket team later.
Dhoni retired from test cricket in 2014, but continued playing in limited overs cricket till 2019.
He has scored 17,266 runs in international cricket including 10,000 plus runs at an average of more than 50 in ODIs.
In 2007, he became the captain of the ODI side before taking over all formats by 2008.
"""

summary = summarizer(text, 3)
print(summary)