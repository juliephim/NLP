
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.metrics import jaccard_distance
nltk.download('wordnet')  # Assurez-vous d'avoir le corpus WordNet téléchargé

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
	#""" Replaces regular expression in a text.
	#>>> replacer = RegexpReplacer()
	#>>> replacer.replace("can't is a contraction")
	#'cannot is a contraction'
	#>>> replacer.replace("I should've done that thing I didn't do")
	#'I should have done that thing I did not do'
	#"""
	def __init__(self, patterns=replacement_patterns):
		self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
	
	def replace(self, text):
		s = text
		
		for (pattern, repl) in self.patterns:
			s = re.sub(pattern, repl, s)
		
		return s
# Fonction pour substitution avant tokenization     
def substitution(text):
    replacer=RegexpReplacer()
    return replacer(text)
     
# Fonction pour tokenizer des phrases
def tokenize_sentences(text):
    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)

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
def remove_stopwords(text, language='english'):
    words = word_tokenize(text)
    stops = set(stopwords.words(language))
    return [word for word in words if word not in stops]

def main():

    # Barre latérale de navigation
    option = st.sidebar.selectbox(
    ('Tokenization', 'Onglet 2', 'Onglet 3'))

    if option == 'Tokenization':
        show_tokenization()
    elif option == 'Onglet 2':
        show_lemmatization()
    else:
        Similarity()

#-----------------------------1er ONGLET : TOKENIZATION WITH NLTK
def show_tokenization():
    st.title("Tokenization with NLTK")
    
    # Saisie de texte par l'utilisateur
    user_input = st.text_area("Enter the text to tokenize:", "")
    
    st.info("For information, the substitution was performing (RegexepReplacer)")

    convert_lower = st.checkbox('Convert to lowercase')
    convert_upper = st.checkbox('Convert to uppercase')

    if convert_lower:
        user_input = user_input.lower()

    if convert_upper:
        user_input = user_input.upper()

    # Choix de la méthode de tokenization
    method = st.selectbox(
        "Select a tokenization method:",
        ("Tokenization of text into sentences", 
         "Tokenization of sentences into words (TreebankWordTokenizer)", 
         "Tokenization by splitting punctuation (WordPunctTokenizer)", 
         "Tokenization using regular expressions(regex)",
         "Removing stop words"))
    
    # Exécution de la tokenization en fonction du choix de l'utilisateur
    if st.button("Tokenizer"):
        if method == "Tokenization of text into sentences":
            st.write(tokenize_sentences(substitution(user_input)))
        elif method == "Tokenization of sentences into words (TreebankWordTokenizer)":
            st.write(tokenize_with_treebank(substitution(user_input)))
        elif method == "Tokenization by splitting punctuation (WordPunctTokenizer)":
            st.write(tokenize_with_wordpunct(substitution(user_input)))
        elif method == "Tokenization using regular expressions(regex)":
            st.write(tokenize_with_regex(substitution(user_input)))
        elif method == "Removing stop words":
            st.write(remove_stopwords(substitution(user_input),'english'))

#-----------------------------2eme ONGLET : Lemmatization & Stemming
def show_lemmatization():
    st.write("Lemmatization & Stemming")

    user_input = st.text_area("Enter the text here:")

    processing_option = st.selectbox(
        'Choose an option:',
        ('Lemmatization', 'Stemming')
    )

    if st.button('Validate'):
        if processing_option == 'Lemmatization':
            lemmatizer = WordNetLemmatizer()
            result = ' '.join([lemmatizer.lemmatize(word) for word in user_input.split()])
            st.write("Result after Lemmatization:", result)
        else:
            stemmer = PorterStemmer()
            result = ' '.join([stemmer.stem(word) for word in user_input.split()])
            st.write("Result after Stemming:", result)

#-----------------------------3eme ONGLET : Similarity_measure
def Similarity():
    st.write("Similarity measurement using Jaccard's coefficient")

    # Les zones de texte pour entrer les séquences de mots
    seq1 = st.text_area("Enter the first sequence of words (separated by spaces):")
    seq2 = st.text_area("Enter the second sequence of words (separated by spaces):")

    if st.button('Calculate Jaccard distance'):
        # Convertir les entrées en ensembles de mots
        X = set(seq1.split())
        Y = set(seq2.split())

        # Calculer la distance de Jaccard
        distance = jaccard_distance(X, Y)
        st.write(f"Jaccard Distance: {distance:.2f}")


if __name__ == "__main__":
    main()
