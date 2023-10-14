
# Importation des bibliothèques nécessaires
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# Fonction pour nettoyer le texte
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprime tout sauf les lettres et les espaces
    text = text.lower()  # Convertit le texte en minuscules
    return text

# Fonction pour vérifier la loi de Zipf
def plot_zipf(text, clean=True):
    if clean:
        text = clean_text(text)
        
    words = text.split()
    word_counts = Counter(words)
    
    # Trie les mots par fréquence
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Récupère les fréquences
    frequencies = [count for word, count in sorted_word_counts]
    
    # Trace le graphique log-log avec modification pour éviter l'avertissement
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(frequencies)
    ax.set_xlabel('Rang')
    ax.set_ylabel('Fréquence')
    ax.set_title('Loi de Zipf')
    st.pyplot(fig)

# Code principal de l'application Streamlit
def main():
    st.title("Vérification de la loi de Zipf")

    user_input = st.text_area("Entrez votre texte ici:")
    clean_option = st.checkbox("Nettoyer le texte (suppression de la ponctuation, conversion en minuscules)", value=True)

    if st.button("Vérifier la loi de Zipf"):
        if user_input:
            plot_zipf(user_input, clean=clean_option)
        else:
            st.write("Veuillez entrer un texte.")

# Si vous exécutez ce script directement, main() sera exécuté.
if __name__ == "__main__":
    main()
