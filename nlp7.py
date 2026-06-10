import nltk

from pypdf import PdfReader

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        content = page.extract_text()

        if content:

            text += content

    return text

documents = []

n = int(input("Enter Number of PDF Files: "))

for i in range(n):

    pdf = input(f"Enter PDF {i+1} Path: ")

    documents.append(
        extract_text(pdf)
    )

vectorizer = CountVectorizer(
    stop_words=stopwords.words('english')
)

X = vectorizer.fit_transform(
    documents
)

topics = int(
    input("Enter Number of Topics: ")
)

lda = LatentDirichletAllocation(
    n_components=topics,
    random_state=42
)

lda.fit(X)

words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):

    print(f"\nTopic {i+1}")

    top_words = topic.argsort()[-5:]

    for index in top_words:

        print(words[index])