import numpy as np
import re
import os
import glob
from collections import defaultdict
import random
from pypdf import PdfReader

random.seed(42)
np.random.seed(42)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def load_documents_from_pdfs(pdf_paths: list[str]) -> tuple[list[str], list[str]]:
    """
    Given a list of PDF file paths, return:
      - documents : list of extracted text strings (one per PDF)
      - names     : list of file basenames for display
    """
    documents, names = [], []
    for path in pdf_paths:
        if not os.path.isfile(path):
            print(f"  [WARN] File not found, skipping: {path}")
            continue
        print(f"  Reading: {os.path.basename(path)} …", end=" ")
        text = extract_text_from_pdf(path)
        if not text:
            print("(no extractable text, skipping)")
            continue
        documents.append(text)
        names.append(os.path.basename(path))
        print(f"({len(text):,} chars)")
    return documents, names



print("=" * 60)
print("  LDA Topic Modelling — PDF Input Mode")
print("=" * 60)
print()
print("How would you like to provide the PDF files?")
print("  1. Enter file paths one by one")
print("  2. Provide a folder and load all PDFs inside it")
choice = input("Choice (1 / 2): ").strip()

pdf_paths: list[str] = []

if choice == "2":
    folder = input("Enter folder path: ").strip().strip('"').strip("'")
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {folder}")
    print(f"Found {len(pdf_paths)} PDF(s) in '{folder}'.")
else:
    n = int(input("Enter the number of PDF files: "))
    for i in range(n):
        path = input(f"  Path to PDF {i + 1}: ").strip().strip('"').strip("'")
        pdf_paths.append(path)

print()
print("Extracting text from PDFs …")
documents, doc_names = load_documents_from_pdfs(pdf_paths)

if not documents:
    raise ValueError("No documents with extractable text were found.")

print(f"\nLoaded {len(documents)} document(s) successfully.")

TOPICS = int(input("\nEnter the number of topics: "))



stop_words = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
}


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return tokens


tokenized_docs = [preprocess(doc) for doc in documents]
vocab = sorted(set(w for doc in tokenized_docs for w in doc))
word2id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
doc_word_ids = [[word2id[w] for w in doc] for doc in tokenized_docs]

print(f"Vocabulary size : {V:,} words")
print(f"Running LDA with {TOPICS} topic(s) for 500 iterations …\n")


class LDA:
    def __init__(self, topics: int, alpha: float = 0.1, beta: float = 0.01,
                 iterations: int = 500):
        self.K = topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def fit(self, docs: list[list[int]], V: int) -> None:
        self.V = V
        D = len(docs)
        self.doc_topic   = np.zeros((D, self.K), dtype=int)
        self.topic_word  = np.zeros((self.K, V), dtype=int)
        self.topic_count = np.zeros(self.K, dtype=int)
        self.assignments = []

        for d, doc in enumerate(docs):
            assigns = []
            for w in doc:
                k = random.randint(0, self.K - 1)
                assigns.append(k)
                self.doc_topic[d][k]  += 1
                self.topic_word[k][w] += 1
                self.topic_count[k]   += 1
            self.assignments.append(assigns)

        # Collapsed Gibbs sampling
        for iteration in range(self.iterations):
            if (iteration + 1) % 100 == 0:
                print(f"  Iteration {iteration + 1}/{self.iterations} …")
            for d, doc in enumerate(docs):
                for i, w in enumerate(doc):
                    k = self.assignments[d][i]
                    self.doc_topic[d][k]  -= 1
                    self.topic_word[k][w] -= 1
                    self.topic_count[k]   -= 1

                    num  = (self.doc_topic[d] + self.alpha) * \
                           (self.topic_word[:, w] + self.beta) / \
                           (self.topic_count + self.V * self.beta)
                    prob = num / num.sum()
                    k_new = np.random.choice(self.K, p=prob)

                    self.assignments[d][i]     = k_new
                    self.doc_topic[d][k_new]  += 1
                    self.topic_word[k_new][w] += 1
                    self.topic_count[k_new]   += 1

        self.phi   = (self.topic_word + self.beta) / \
                     (self.topic_count[:, None] + self.V * self.beta)
        self.theta = (self.doc_topic + self.alpha) / \
                     (self.doc_topic + self.alpha).sum(axis=1, keepdims=True)

    def top_words(self, n: int = 8) -> list[list[str]]:
        return [
            [vocab[i] for i in self.phi[k].argsort()[::-1][:n]]
            for k in range(self.K)
        ]

    def doc_dominant_topic(self) -> np.ndarray:
        return self.theta.argmax(axis=1)



lda = LDA(topics=TOPICS, alpha=0.1, beta=0.01, iterations=500)
lda.fit(doc_word_ids, V)

top_words_list      = lda.top_words(n=8)
auto_topic_labels   = [words[0].upper() for words in top_words_list]
dominant_topics     = lda.doc_dominant_topic()

print("\n" + "=" * 60)
print("  TOPICS DISCOVERED")
print("=" * 60)
for k in range(TOPICS):
    print(f"\nTopic {k + 1}  [Label: {auto_topic_labels[k]}]")
    print(f"  Keywords : {', '.join(top_words_list[k])}")

print("\n" + "=" * 60)
print("  DOCUMENT CLUSTERS")
print("=" * 60)

clusters: dict[int, list[int]] = defaultdict(list)
for d, topic in enumerate(dominant_topics):
    clusters[topic].append(d)

for k in range(TOPICS):
    if clusters[k]:
        print(f"\nCluster: {auto_topic_labels[k]}")
        for d in clusters[k]:
            snippet = documents[d][:200].replace('\n', ' ')
            print(f"  [{d + 1:02d}] {doc_names[d]}")
            print(f"       {snippet} …")