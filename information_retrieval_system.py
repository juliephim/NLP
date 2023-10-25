
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


def loadCNN():
    file = open("CNNArticles",'rb')
    articles = pickle.load(file)
    file = open("CNNGold",'rb')
    abstracts = pickle.load(file)

    articlesCl = []  
    for article in articles:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    articles = articlesCl
	  
    articlesCl = []  
    for article in abstracts:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    abstracts = articlesCl
    
    return articles, abstracts


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)


def information_retrieval_interface(query, vectorizer, tfidf_articles, articles):
    print("Traitement de votre requête...")
    
    preprocessed_query = preprocess(query)
    tfidf_query = vectorizer.transform([preprocessed_query])
    similarities = cosine_similarity(tfidf_query, tfidf_articles)
    ranked_indices = similarities.argsort()[0][::-1]
    
    print("\nVoici la liste des articles classés selon leur pertinence avec votre requête:\n")
    for i, index in enumerate(ranked_indices[:5], 1):
        print(f"Article {i}:\n{articles[index]}\n{'='*50}\n")
    
    return [articles[i] for i in ranked_indices]


# Chargement des données
articles, abstracts = loadCNN()

# Prétraitement des données
preprocessed_articles = [preprocess(article) for article in articles[:100]]
preprocessed_abstracts = [preprocess(abstract) for abstract in abstracts[:100]]

# Vectorisation avec TF-IDF
all_texts = preprocessed_articles + preprocessed_abstracts
vectorizer = TfidfVectorizer()
vectorizer.fit(all_texts)
tfidf_articles = vectorizer.transform(preprocessed_articles)

# Exemple d'utilisation de l'interface
sample_query = abstracts[101]
information_retrieval_interface(sample_query, vectorizer, tfidf_articles, articles)
