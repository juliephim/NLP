# Import necessary libraries for data handling and text processing.
#import urllib.request as re
import numpy as np
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi

import string

# Download the stopwords dataset and the punkt tokenizer model from NLTK, which will be used for tokenization and stop word removal.
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')


# Define a function to load the NFCorpus data from the cloned GitHub repository.
def loadNFCorpus():

    # Define the directory where the data is located.
    dir = "./project1-2023/"
    # Load the document data which contains abstracts from PubMed.
    filename = dir + "dev.docs"
    
    # Initialize a dictionary to store document data.
    dicDoc = {}
    # Read document lines and split them into a dictionary with key as document ID and value as text.
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        tabLine = line.split('\t')
        key = tabLine[0]
        value = tabLine[1]
        dicDoc[key] = value
    
    # Load and parse the query data similar to document data.
    filename = dir + "dev.all.queries"
    dicReq = {}
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        tabLine = line.split('\t')
        key = tabLine[0]
        value = tabLine[1]
        dicReq[key] = value
    
    # Load the relevance judgments which provide a relevance score for document-query pairs.
    filename = dir + "dev.2-1-0.qrel"
    dicReqDoc = defaultdict(dict)
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        tabLine = line.strip().split('\t')
        req = tabLine[0]
        doc = tabLine[2]
        score = int(tabLine[3])
        dicReqDoc[req][doc] = score
    
    # Return the loaded document and query data along with relevance judgments.
    return dicDoc, dicReq, dicReqDoc

def text2TokenList(text):
    # Define stop words
    stop_words = set(stopwords.words('english'))
    # Initialize tokenizer and lemmatizer
    tokenizer = TreebankWordTokenizer()
    #tokenizer =RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    # Remove punctuation from text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text using TreebankWordTokenizer
    word_tokens = tokenizer.tokenize(text.lower())
    # Lemmatization with POS tagging
    word_tokens_lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens]
    # Remove stopwords and filter out short tokens
    word_tokens_final = [word for word in word_tokens_lemmatized if word not in stop_words and len(word) > 2]
    return word_tokens_final


def text3TokenList(text, top_n=27):
    # First, get word-level tokens
    word_tokens = text2TokenList(text)

    # Next, extract key phrases or concepts
    # This example uses TF-IDF to extract top N terms
    vectorizer = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

    top_n_words = feature_array[tfidf_sorting][:top_n]
    
    # Combine word tokens and top N terms
    combined_tokens = word_tokens + list(top_n_words)
    return combined_tokens

# Function to vectorize text using Word2Vec
def vectorize_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# [Place your code for loading the NFCorpus here]
dicDoc, dicReq, dicReqDoc = loadNFCorpus()

# Prepare the corpus for Word2Vec training
corpus_for_word2vec = [text2TokenList(doc) for doc in dicDoc.values()]

# Train the Word2Vec model
word2vec_model = Word2Vec(corpus_for_word2vec, vector_size=100, window=7, min_count=1, epochs=10, workers=4)

# Define a function to combine BM25 and Word2Vec scores
from sklearn.metrics.pairwise import cosine_similarity
def combined_score(bm25_scores, doc_vectors, query_vector, alpha=0.2):
    cosine_similarities = [cosine_similarity([dv], [query_vector])[0][0] for dv in doc_vectors]
    combined_scores = alpha * np.array(bm25_scores) + (1 - alpha) * np.array(cosine_similarities)
    return combined_scores

# Define a function to execute the BM25 ranking model on a subset of the NFCorpus.
def run_bm25_with_word2vec(startDoc, endDoc):
    dicDoc, dicReq, dicReqDoc = loadNFCorpus()
    # Initialize lists to keep track of documents and queries used in the experiment.
    docsToKeep = []
    reqsToKeep = []
    dicReqDocToKeep = defaultdict(dict)

    # Set the number of top documents to evaluate with ndcg (normalized discounted cumulative gain).
    ndcgTop = 5

    # Iterate over the document-query relevance data to create a subset for the experiment.
    i = startDoc
    for reqId in dicReqDoc:
        if i > (endDoc - startDoc):
            break
        for docId in dicReqDoc[reqId]:
            dicReqDocToKeep[reqId][docId] = dicReqDoc[reqId][docId]
            docsToKeep.append(docId)
            i = i + 1
        reqsToKeep.append(reqId)
    docsToKeep = list(set(docsToKeep))

    # Create a corpus of token lists for documents and queries, along with their IDs.
    corpusDocTokenList = []
    corpusReqTokenList = {}
    corpusDocName = []
    corpusDicoDocName = {}
    i = 0
    for k in docsToKeep:
        docTokenList = text3TokenList(dicDoc[k])
        corpusDocTokenList.append(docTokenList)
        corpusDocName.append(k)
        corpusDicoDocName[k] = i
        i = i + 1

    # Create a dictionary to map queries to their token lists.
    corpusReqName = []
    corpusDicoReqName = {}
    i = 0
    for k in reqsToKeep:
        reqTokenList = text3TokenList(dicReq[k])
        corpusReqTokenList[k] = reqTokenList
        corpusReqName.append(k)
        corpusDicoReqName[k] = i
        i = i + 1

    # Vectorize documents using Word2Vec
    doc_vectors = [vectorize_text(doc, word2vec_model) for doc in corpusDocTokenList]

    # Initialize BM25 with the document token lists
    bm25 = BM25Okapi(corpusDocTokenList)

    # Calculate the cumulative NDCG score for all queries
    ndcgCumul = 0
    nbReq = 0
    for req in corpusReqTokenList:
        reqTokenList = corpusReqTokenList[req]
        bm25_scores = bm25.get_scores(reqTokenList)
        query_vector = vectorize_text(reqTokenList, word2vec_model)
        trueDocs = np.zeros(len(corpusDocTokenList))

        combined_scores = combined_score(bm25_scores, doc_vectors, query_vector)
        
        # Create an array to mark the relevance of the true documents as per relevance judgments.
        for docId in corpusDicoDocName:
            if req in dicReqDocToKeep and docId in dicReqDocToKeep[req]:
                posDocId = corpusDicoDocName[docId]
                trueDocs[posDocId] = dicReqDocToKeep[req][docId]

        # Update the cumulative NDCG score with the scores from the current query.
        ndcgCumul = ndcgCumul + ndcg_score([trueDocs], [combined_scores],k=ndcgTop)
        nbReq = nbReq + 1
    ndcgCumul = ndcgCumul / nbReq
    # Print final NDCG score
    print("Combined NDCG score =", ndcgCumul)
    return ndcgCumul

# Run the combined model
nb_docs = 3192 # Adjust as needed
run_bm25_with_word2vec(0, nb_docs)