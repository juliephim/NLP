import streamlit as st
import string
import re
import nltk
from os import listdir
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from pickle import dump


nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Load document into memory
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

# Turn a document into clean tokens
def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# Extract adverbs using POS tagging
def extract_adverbs(tokens):
    pos_tags = nltk.pos_tag(tokens)
    adverbs = [word for word, tag in pos_tags if tag in ('RB', 'RBR', 'RBS')]
    return adverbs

# Load all docs in a directory and extract adverbs
def process_docs(directory, is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        adverbs = extract_adverbs(tokens)
        documents.append(' '.join(adverbs))
    return documents

# Load, clean, and extract adverbs from a dataset
def load_clean_dataset(is_train):
    neg = process_docs('./neg', is_train)
    pos = process_docs('./pos', is_train)
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def get_sentiment_score(word):
    """Get the sentiment score of a word using SentiWordNet."""
    synsets = wn.synsets(word, pos=wn.ADV)
    
    # If no synset is found, return a neutral score
    if not synsets:
        return 0
    
    # Get the first synset (most common sense)
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    
    # Compute a sentiment score: positivity - negativity
    sentiment_score = swn_synset.pos_score() - swn_synset.neg_score()
    
    return sentiment_score

def score_reviews(reviews):
    """Score reviews based on the sentiment of their adverbs."""
    scores = []
    
    for review in reviews:
        adverbs = review.split()
        if not adverbs:  # If no adverbs in the review, consider it neutral
            scores.append(0)
            continue
        
        total_score = sum(get_sentiment_score(adverb) for adverb in adverbs)
        avg_score = total_score / len(adverbs)
        scores.append(avg_score)
    
    return scores

def classify_reviews(scores, pos_threshold=0.2, neg_threshold=-0.2):
    """Classify reviews based on their sentiment scores."""
    classifications = []
    
    for score in scores:
        if score > pos_threshold:
            classifications.append("Positive")
        elif score < neg_threshold:
            classifications.append("Negative")
        else:
            classifications.append("Neutral")
    
    return classifications

def main():
    st.title("Movie Review Sentiment Analysis")

    # Text area for user input
    review = st.text_area("Enter the movie review:")

    # Button to process the review
    if st.button("Analyze"):
        if review:
            # Process and score the user's review
            processed_review = process_single_review(review)
            sentiment_score = score_reviews([processed_review])[0]
            classification = classify_reviews([sentiment_score])[0]
            
            # Display the results
            st.write(f"Processed Review (Adverbs): {processed_review}")
            st.write(f"Sentiment Score: {sentiment_score}")
            st.write(f"Classification: {classification}")
            st.write("Legend:")
            st.write("Positive: Score > 0.2")
            st.write("Neutral: Score between -0.2 and 0.2")
            st.write("Negative: Score < -0.2")
        else:
            st.write("Please enter a review to analyze.")

if __name__ == '__main__':
    main()


#### CODE TO NOT INCLUDE


# Sample movie review
sample_review = """The film was brilliantly directed and the actors performed superbly. 
                   However, the plot moved incredibly slowly and the ending was sadly predictable."""

# Function to process a single review
def process_single_review(review):
    tokens = clean_doc(review)
    adverbs = extract_adverbs(tokens)
    return ' '.join(adverbs)

# Process and score the sample review
processed_review = process_single_review(sample_review)
sentiment_score = score_reviews([processed_review])[0]
classification = classify_reviews([sentiment_score])[0]

#-----------------------------------
# Print the results
print(f"Sample Review: {sample_review}\n")
print(f"Processed Review (Adverbs): {processed_review}\n")
print(f"Sentiment Score: {sentiment_score}\n")
print(f"Classification: {classification}\n")



#Code qui voit les performances de notre classifier avec le dataset polarity movie reviews
# Load the dataset
reviews, labels = load_clean_dataset(is_train=True)
# Score the reviews
sentiment_scores = score_reviews(reviews)
# Classify the reviews based on their scores
classifications = classify_reviews(sentiment_scores)
# Convert classifications into numerical labels: Positive = 1, Negative = 0, Neutral = -1 (we'll ignore Neutral for accuracy)
predicted_labels = [1 if c == "Positive" else 0 if c == "Negative" else -1 for c in classifications]
# Remove Neutral reviews for accuracy calculation
filtered_labels = [label for i, label in enumerate(labels) if predicted_labels[i] != -1]
filtered_predictions = [pred for pred in predicted_labels if pred != -1]
# Compute accuracy
correct_predictions = sum(1 for true, pred in zip(filtered_labels, filtered_predictions) if true == pred)
accuracy = correct_predictions / len(filtered_labels)
# Print the accuracy
print(f"For comparison, the classifier have a score of {accuracy * 100:.2f}% for the dataset polarity dataset.")

# code pour voir les 5 premiers reviews de polarity movie review dataset, en comparaison avec notre classification basé sur le pos_tag et sentiwordnet
# Load the dataset
reviews, labels = load_clean_dataset(is_train=True)
# Score the reviews
sentiment_scores = score_reviews(reviews)
# Classify the reviews based on their scores
classifications = classify_reviews(sentiment_scores)
# Print the first few reviews and their classifications
for review, label, classification in zip(reviews[:17],labels[:17], classifications[:17]):
    print(f"Review: {review}\n Labels:{label} Classification: {classification}\n")
