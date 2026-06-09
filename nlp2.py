import nltk
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = input("Enter Text: ")

words = word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))

words = [w for w in words if w.isalpha() and w not in stop_words]

freq = Counter(words)

print(freq)

wc = WordCloud().generate(" ".join(words))

plt.imshow(wc)
plt.axis("off")
plt.show()