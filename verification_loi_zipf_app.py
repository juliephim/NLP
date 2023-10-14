
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re


# Code for handling contractions

replacement_patterns = [
	(r'won\'t', 'will not'),
	(r'can\'t', 'cannot'),
	(r'i\'m', 'i am'),
	(r'ain\'t', 'is not'),
	(r'(\w+)\'ll', '\g<1> will'),
	(r'(\w+)n\'t', '\g<1> not'),
	(r'(\w+)\'ve', '\g<1> have'),
	(r'(\w+)\'s', '\g<1> is'),
	(r'(\w+)\'re', '\g<1> are'),
	(r'(\w+)\'d', '\g<1> would'),
]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


# Code for language detection

from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Indéterminé"


# Fonction pour nettoyer le texte
def clean_text(text):
    replacer = RegexpReplacer()
    text = replacer.replace(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Fonction pour vérifier la loi de Zipf
def verify_zipf_law(text):
    words = text.split()
    word_freq = Counter(words)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    frequencies = [freq for word, freq in sorted_word_freq]
    plt.figure(figsize=(10, 7))
    plt.loglog(frequencies, linestyle='-', marker='o')
    plt.ylabel('Fréquence')
    plt.xlabel('Rang')
    plt.title('Vérification de la loi de Zipf')
    plt.grid(True)
    st.pyplot(plt)

# Interface Streamlit
st.title("Vérification de la loi de Zipf")
st.write(
    "Zipf's law states that the frequency of a token in a text is directly proportional to its rank or position in the sorted list.",
    "This means that the most common word will appear approximately x2 (twice) as often as the second most common word, x3 times as often as the third most common word, and so on.",
    "This distribution reflects a consistent pattern in many large texts, demonstrating the predictability of word frequencies across diverse languages and genres.",
    "The Zipf's law suggests that extremely common and extremely rare words are often less informative in themselves. Words that fall in the middle of the distribution, neither too common nor too rare, tend to be the most informative."
)
text_input = st.text_area("Entrez votre texte ici:")

# Language detection and message display
lang_detected = detect_language(text_input)
if lang_detected == "en":
    st.write("Ohh! I guessed that the text is in English. Let's see if Zipf's law can be applied!! :)")
elif lang_detected == "fr":
    st.write("Ohh! I guessed that the text is in French. Let's see if Zipf's law can be applied!! :)")
else:
    st.write("Sorry, I couldn't determine the language. Let's see if Zipf's law can be applied!! :)")

if st.button("Vérifier"):
    cleaned_text = clean_text(text_input)
    verify_zipf_law(cleaned_text)
