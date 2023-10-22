
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer, TreebankWordTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords

# Fonction pour tokenizer des phrases
def tokenize_sentences(text):
    return sent_tokenize(text)

# Fonction pour tokenizer des mots avec TreebankWordTokenizer
def tokenize_with_treebank(text):
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)

# Fonction pour tokenizer des mots avec WordPunctTokenizer
def tokenize_with_wordpunct(text):
    tokenizer = WordPunctTokenizer()
    return tokenizer.tokenize(text)

# Fonction pour tokenizer avec des expressions régulières
def tokenize_with_regex(text):
    tokenizer = RegexpTokenizer("\w+")
    return tokenizer.tokenize(text)

# Fonction pour supprimer les mots vides (stop words)
def remove_stopwords(text, language='french'):
    words = word_tokenize(text)
    stops = set(stopwords.words(language))
    return [word for word in words if word not in stops]

def main():
    st.title("Tokenization avec NLTK")
    
    # Saisie de texte par l'utilisateur
    user_input = st.text_area("Entrez le texte à tokenizer:", "")
    
    # Choix de la méthode de tokenization
    method = st.selectbox(
        "Sélectionnez une méthode de tokenization:",
        ("Tokenization de phrases", 
         "Tokenization de mots (TreebankWordTokenizer)", 
         "Tokenization avec ponctuation (WordPunctTokenizer)", 
         "Tokenization avec expressions régulières",
         "Suppression des mots vides (stop words)"))
    
    # Exécution de la tokenization en fonction du choix de l'utilisateur
    if st.button("Tokenizer"):
        if method == "Tokenization de phrases":
            st.write(tokenize_sentences(user_input))
        elif method == "Tokenization de mots (TreebankWordTokenizer)":
            st.write(tokenize_with_treebank(user_input))
        elif method == "Tokenization avec ponctuation (WordPunctTokenizer)":
            st.write(tokenize_with_wordpunct(user_input))
        elif method == "Tokenization avec expressions régulières":
            st.write(tokenize_with_regex(user_input))
        elif method == "Suppression des mots vides (stop words)":
            st.write(remove_stopwords(user_input))

if __name__ == "__main__":
    main()
