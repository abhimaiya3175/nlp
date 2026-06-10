from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


print("\n------ TEXT SUMMARIZER USING TF-IDF ------")
text = input("\nEnter Paragraph/Text:\n")
length = int(input("\nEnter Number of Sentences in Summary: "))


sentences = sent_tokenize(text)


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sentences)
scores = tfidf_matrix.sum(axis=1)
scores = scores.A1


ranked_sentences = sorted(
    zip(sentences, scores),
    key=lambda x: x[1],
    reverse=True
)
summary = " ".join(
    sentence
    for sentence, score in ranked_sentences[:length]
)
print("\n------ GENERATED SUMMARY ------\n")
print(summary)
