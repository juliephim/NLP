
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords
import re


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
    st.title("Tokenization avec NLTK")
    
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

if __name__ == "__main__":
    main()
