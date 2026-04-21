import matplotlib.pyplot as plt
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    words = word_tokenize(text.lower()) 
    s_w = set(stopwords.words('english'))
    dictionary = {}
    
    for word in words:
        if word == 'extensions':
            word = 'extension'
            
        if word.isalpha() and word not in s_w:
            dictionary[word] = dictionary.get(word, 0) + 1
            
    return dictionary

def wordcloud(freq):
    fig, ax = plt.subplots(figsize=(8, 8))
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    placed_bboxes = []
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for key, val in sorted_freq:
        fontsize = 12 + (val * 10)
        
        for _ in range(500):
            x, y = random.uniform(0.1, 0.8), random.uniform(0.1, 0.8)
            color = (random.random() * 0.7, random.random() * 0.7, random.random() * 0.7)
            
            t = ax.text(x, y, key, fontsize=fontsize, color=color, 
                        fontweight='bold', ha='center', va='center')
            
            bbox = t.get_window_extent(renderer=renderer)
            
            if any(bbox.overlaps(p_bbox2) for p_bbox in placed_bboxes):
                t.remove()
            else:
                placed_bboxes.append(bbox)
                break

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()

text = """Read the migration plan to Notebook 7 to learn about the new features and the actions to take if you are using extensions - Please note that updating to Notebook 7 might break some of your extensions. Dhoni retired from test cricket in 2014, but continued playing in limited overs cricket till 2019. He has scored 17,266 runs in international cricket including 10,000 plus runs at an average of more than 50 in ODI. In 2007, he became the captain of the ODI side before taking over in all formats by 2008"""

result = preprocess(text)
wordcloud(result)