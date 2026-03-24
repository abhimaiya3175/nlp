import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

print(nltk.data.find('tokenizers/punkt'))

def tf_idf(sentence, sentences):
    # preprocess the text 
    def preprocess(text):
        text = text.lower()
        cleaned = []
        words = word_tokenize(text)
        s_w = set(stopwords.words('english'))

        for word in words:
            if word not in s_w:
                if word.isalnum():
                    cleaned.append(word)
        return cleaned

    # calculate tf
    def tf(word, sentence):
        words = word_tokenize(sentence)
        count = 0
        for w in words:
            if word == w:
                count += 1
        return count / len(words) if len(words) > 0 else 0

    # calculate idf
    def idf(word, sentences):
        count = 0
        for sent in sentences:
            if word in word_tokenize(sent):   # fixed token check
                count += 1
        return np.log(len(sentences) / (count + 1))

    words = preprocess(sentence)
    tf_idf_score = 0

    for word in words:
        tf_score = tf(word, sentence)
        idf_score = idf(word, sentences)
        tf_idf_score += tf_score * idf_score

    return tf_idf_score


# FIXED: Proper indentation (outside tf_idf)
def summarizer(text, length):
    sent_score = {}
    sentences = sent_tokenize(text)

    for sentence in sentences:
        score = tf_idf(sentence, sentences)
        sent_score[sentence] = score
    
    sorted_sent_score = dict(sorted(sent_score.items(), key=lambda x: x[1], reverse=True))
    top_k = list(sorted_sent_score.keys())[:length]
    
    summary = ' '.join(top_k)
    return summary


# tf-idf sentence summary
text = """
Born in Ranchi, Dhoni made his first class debut for Bihar in 1999. He made his debut for the Indian cricket team on 23 December 2004 in an ODI against Bangladesh and played his first test a year later against Sri Lanka. In 2007, he became the captain of the ODI side before taking over in all formats by 2008. Dhoni retired from test cricket in 2014, but continued playing in limited overs cricket till 2019. He has scored 17,266 runs in international cricket including 10,000 plus runs at an average of more than 50 in ODIs.
"""

summary = summarizer(text, 3)
print(summary)