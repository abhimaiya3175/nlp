import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np

# ---------------- DOWNLOAD NLTK DATA ---------------- #

nltk.download('punkt')
nltk.download('stopwords')

# ---------------- PREPROCESS FUNCTION ---------------- #

def preprocess(text):

    text = text.lower()

    cleaned = []

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    for word in words:

        if word.isalnum() and word not in stop_words:

            cleaned.append(word)

    return cleaned

# ---------------- TF FUNCTION ---------------- #

def tf(word, sentence):

    words = word_tokenize(sentence.lower())

    count = 0

    for w in words:

        if w == word:

            count += 1

    if len(words) == 0:

        return 0

    return count / len(words)

# ---------------- IDF FUNCTION ---------------- #

def idf(word, sentences):

    count = 0

    for sent in sentences:

        words = word_tokenize(sent.lower())

        if word in words:

            count += 1

    return np.log(len(sentences) / (count + 1))

# ---------------- TF-IDF SCORE ---------------- #

def tf_idf(sentence, sentences):

    words = preprocess(sentence)

    score = 0

    for word in words:

        tf_score = tf(word, sentence)

        idf_score = idf(word, sentences)

        score += tf_score * idf_score

    return score

# ---------------- SUMMARIZER FUNCTION ---------------- #

def summarizer(text, length):

    sentence_scores = {}

    sentences = sent_tokenize(text)

    for sentence in sentences:

        score = tf_idf(sentence, sentences)

        sentence_scores[sentence] = score

    # sort sentences based on score
    sorted_sentences = sorted(
        sentence_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_sentences = []

    for i in range(min(length, len(sorted_sentences))):

        top_sentences.append(sorted_sentences[i][0])

    summary = " ".join(top_sentences)

    return summary

# ---------------- USER INPUT ---------------- #

print("\n------ TEXT SUMMARIZER USING TF-IDF ------")

text = input("\nEnter Paragraph/Text:\n")

length = int(input("\nEnter Number of Sentences in Summary: "))

# ---------------- GENERATE SUMMARY ---------------- #

summary = summarizer(text, length)

# ---------------- OUTPUT ---------------- #

print("\n------ GENERATED SUMMARY ------\n")

print(summary)