import os
import re
import random
import numpy as np
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS=set(stopwords.words('english'))

def extract_text_from_pdf(pdf_path):
    text=""
    try:
        reader=PdfReader(pdf_path)
        for page in reader.pages:
            content=page.extract_text()
            if content:
                text+=content+" "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def load_pdfs(folder_path):
    documents=[]
    filenames=[]
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path=os.path.join(folder_path,file)
            text=extract_text_from_pdf(full_path)
            documents.append(text)
            filenames.append(file)
    return documents,filenames

def preprocess(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]',' ',text)
    words=word_tokenize(text)
    words=[w for w in words if w not in STOP_WORDS and len(w)>2]
    return words

def build_vocabulary(docs):
    vocab={}
    index=0
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word]=index
                index+=1
    return vocab

class LDAModel:
    def __init__(self,num_topics=3,alpha=0.1,beta=0.01,iterations=100):
        self.K=num_topics
        self.iterations=iterations
        self.alpha=alpha
        self.beta=beta

    def fit(self,docs,vocab):
        self.vocab=vocab
        self.id2word={i:w for w,i in vocab.items()}

        self.docs=[[vocab[w] for w in doc if w in vocab] for doc in docs]

        D=len(self.docs)
        V=len(vocab)

        self.doc_topic=np.zeros((D,self.K))
        self.topic_word=np.zeros((self.K,V))
        self.topic_total=np.zeros(self.K)

        self.assignments=[]

        for d,doc in enumerate(self.docs):
            topics=[]
            for w in doc:
                k=random.randint(0,self.K-1)
                topics.append(k)
                self.doc_topic[d][k]+=1
                self.topic_word[k][w]+=1
                self.topic_total[k]+=1
            self.assignments.append(topics)

        for _ in range(self.iterations):
            for d,doc in enumerate(self.docs):
                for i,w in enumerate(doc):

                    old=self.assignments[d][i]

                    self.doc_topic[d][old]-=1
                    self.topic_word[old][w]-=1
                    self.topic_total[old]-=1

                    probs=[]
                    for k in range(self.K):
                        p=((self.topic_word[k][w]+self.beta)/
                           (self.topic_total[k]+V*self.beta)) * \
                          (self.doc_topic[d][k]+self.alpha)
                        probs.append(p)

                    probs=np.array(probs)
                    probs/=probs.sum()

                    new=np.random.choice(self.K,p=probs)

                    self.assignments[d][i]=new
                    self.doc_topic[d][new]+=1
                    self.topic_word[new][w]+=1
                    self.topic_total[new]+=1

    def print_topics(self,top_n=10):
        for k in range(self.K):
            top=self.topic_word[k].argsort()[-top_n:][::-1]
            words=[self.id2word[i] for i in top]
            print(f"\nTopic {k+1}:")
            print(", ".join(words))

    def get_document_topics(self):
        return np.argmax(self.doc_topic,axis=1)

folder_path=input("Enter PDF folder path: ")
num_topics=int(input("Enter number of topics: "))

raw_docs,filenames=load_pdfs(folder_path)

if len(raw_docs)==0:
    print("No PDFs found!")

processed_docs=[preprocess(doc) for doc in raw_docs]

vocab=build_vocabulary(processed_docs)

print("Vocabulary Size:",len(vocab))

lda=LDAModel(num_topics=num_topics,iterations=100)

lda.fit(processed_docs,vocab)

lda.print_topics(top_n=10)

doc_topics=lda.get_document_topics()

topic_to_files={i:[] for i in range(num_topics)}

for i,t in enumerate(doc_topics):
    topic_to_files[t].append(filenames[i])

print("\n===== DOCUMENT CLUSTERS =====")

for t in range(num_topics):
    print(f"\nTopic {t+1}:")
    if len(topic_to_files[t])==0:
        print("No PDFs assigned")
    else:
        for f in topic_to_files[t]:
            print("-",f)
