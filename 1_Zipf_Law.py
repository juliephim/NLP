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



def tokenize_text(text, lang="en"):
    """
    Tokenizes the text using nltk's word_tokenize.
    Args:
    - text (str): Input text.
    - lang (str): Language of the text (default is "en").
    
    Returns:
    List[str]: List of tokens.
    """
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if lang in ["en", "fr"]:
        stop_words = set(stopwords.words(lang))
        tokens = [word for word in tokens if word.lower() not in stop_words]
    
    return tokens


# Fonction pour nettoyer le texte
def corrected_clean_text(text, lang="en"):
    replacer = RegexpReplacer()
    text = replacer.replace(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # If language is French, use a regex pattern that conserves French accents
    if lang == "fr":
        text = re.sub(r'[^a-zàâäéèêëîïôöùûüçA-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ\s]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
    return text

# Fonction pour vérifier la loi de Zipf
def updated_verify_zipf_law(text, lang="en"):
    words = tokenize_text(text, lang)
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


import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# Charger les modèles spaCy pour l'anglais et le français
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Créer une fonction pour la lemmatisation
def lemmatize_text(text, language="en"):
    if language == "en":
        doc = nlp_en(text)
    elif language == "fr":
        doc = nlp_fr(text)
    else:
        return text  # Si la langue n'est ni anglais ni français, retourner le texte tel quel
    return " ".join([token.lemma_ for token in doc])

# Créer une fonction pour la tokenisation
def tokenize_text(text, remove_stopwords=False, language="en"):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    if remove_stopwords:
        if language == "en":
            stop_words = set(stopwords.words('english'))
        elif language == "fr":
            stop_words = set(stopwords.words('french'))
        else:
            stop_words = set()
        tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Language detection and message display
lang_detected = detect_language(text_input)
if lang_detected == "en":
    st.write("Ohh! I guessed that the text is in English. Let's see if Zipf's law can be applied!! :)")
elif lang_detected == "fr":
    st.write("Ohh! I guessed that the text is in French. Let's see if Zipf's law can be applied!! :)")
else:
    st.write("Sorry, I couldn't determine the language. Let's see if Zipf's law can be applied!! :)")


remove_stopwords = st.checkbox("Supprimer les mots vides")
if st.button("Vérifier"):
    detected_lang = detect_language(text_input)
    cleaned_text = corrected_clean_text(text_input, detected_lang)
    tokenized_text = tokenize_text(cleaned_text, remove_stopwords, lang_detected)
    lemmatized_text = lemmatize_text(tokenized_text, lang_detected)
    st.write("Texte après tokenisation et prétraitement :")
    st.write(lemmatized_text)
    detected_lang = detect_language(text_input)
    cleaned_text = corrected_clean_text(text_input, detected_lang)
    updated_verify_zipf_law(cleaned_text, detected_lang)

