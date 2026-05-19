import matplotlib.pyplot as plt
import random
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------- DOWNLOAD NLTK DATA ---------------- #

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------- LEMMATIZER ---------------- #

lemmatizer = WordNetLemmatizer()

# ---------------- PREPROCESS FUNCTION ---------------- #

def preprocess(text):

    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))

    dictionary = {}

    for word in words:

        # remove stopwords and symbols
        if word.isalpha() and word not in stop_words:

            # lemmatization
            word = lemmatizer.lemmatize(word)

            dictionary[word] = dictionary.get(word, 0) + 1

    return dictionary

# ---------------- WORD CLOUD FUNCTION ---------------- #

def wordcloud(freq):

    fig, ax = plt.subplots(figsize=(10, 10))

    # sort words based on frequency
    sorted_freq = sorted(
        freq.items(),
        key=lambda x: x[1],
        reverse=True
    )

    placed_bboxes = []

    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()

    for word, count in sorted_freq:

        # font size based on frequency
        fontsize = 12 + (count * 10)

        for _ in range(500):

            # random position
            x = random.uniform(0.1, 0.9)

            y = random.uniform(0.1, 0.9)

            # random color
            color = (
                random.random() * 0.7,
                random.random() * 0.7,
                random.random() * 0.7
            )

            t = ax.text(
                x,
                y,
                word,
                fontsize=fontsize,
                color=color,
                fontweight='bold',
                ha='center',
                va='center'
            )

            bbox = t.get_window_extent(renderer=renderer)

            # avoid overlapping
            overlap = False

            for placed_bbox in placed_bboxes:

                if bbox.overlaps(placed_bbox):

                    overlap = True

                    break

            if overlap:

                t.remove()

            else:

                placed_bboxes.append(bbox)

                break

    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1)

    ax.axis('off')

    plt.title("Custom Word Cloud with Lemmatization")

    plt.show()

# ---------------- USER INPUT ---------------- #

print("\n------ CUSTOM WORD CLOUD GENERATOR ------")

text = input("\nEnter Paragraph/Text:\n")

# ---------------- PROCESS TEXT ---------------- #

result = preprocess(text)

# ---------------- DISPLAY WORD FREQUENCY ---------------- #

print("\nWord Frequencies:\n")

for word, count in result.items():

    print(word, ":", count)

# ---------------- GENERATE WORD CLOUD ---------------- #

wordcloud(result)