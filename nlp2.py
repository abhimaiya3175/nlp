import matplotlib.pyplot as plt
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    words = word_tokenize(text.lower())  # convert to lowercase
    s_w = set(stopwords.words('english'))

    dictionary = {}

    for word in words:
        if word not in s_w and word.isalnum():
            dictionary[word] = dictionary.get(word, 0) + 1

    return dictionary


def wordcloud(freq):
    plt.figure(figsize=(8, 8))

    # sort dictionary by frequency
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    grid_size = 10
    grid = [[False]*grid_size for _ in range(grid_size)]

    for key, val in freq:
        x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)

        # avoid overlap
        while grid[x][y]:
            x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)

        grid[x][y] = True

        color = (random.random(), random.random(), random.random())

        plt.text(x/grid_size, y/grid_size, key,
                 fontsize=10 + val*5, color=color)

    plt.axis('off')
    plt.title("Word Cloud")
    plt.show()


text = """On 14 November 1987, at age 14, Tendulkar was selected to represent Bombay..."""

processed_text = preprocess(text)
wordcloud(processed_text)