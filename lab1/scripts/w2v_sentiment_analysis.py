import glob
import os
import re
import sys
from gensim.models import Word2Vec, KeyedVectors

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from w2v_train import W2VLossLogger

SCRIPT_DIRECTORY = os.path.realpath(__file__)

data_dir = 'data/aclImdb/' 
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)

def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)

def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())

def tokenize(s):
    return s.split(" ")

def preproc_tok(s):
    return tokenize(preprocess(s))

def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r") as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data

def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])

def extract_nbow(w2v, corpus):
    """Extract neural bag of words representations"""
    nbow_corpus = []
    for sentence in corpus:
        vec = np.array(w2v.vector_size*[0.])
        for cnt, word in enumerate(sentence):
            try:
                vec += np.array(w2v[word])
            except:
                vec += np.array(w2v.vector_size*[0.])
        nbow_corpus.append(vec/(cnt+1))
    return nbow_corpus

def train_sentiment_analysis(train_corpus, train_labels):
    #"""Train a sentiment analysis classifier using NBOW + Logistic regression"""
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_corpus, train_labels)
    return classifier


def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
   # """Evaluate classifier on the test corpus and report accuracy, precision, and recall"""
    predictions = classifier.predict(test_corpus)
    report = classification_report(test_labels, predictions, digits=4)
    
    print(report)
    
    return report

if __name__ == "__main__":
    
    w2v_gut = Word2Vec.load('gutenberg_w2v.100d.model').wv
    w2v_google = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=NUM_W2V_TO_LOAD)

    pos_train = read_samples(pos_train_dir, preprocess=preproc_tok)
    pos_test = read_samples(pos_test_dir, preprocess=preproc_tok)
    neg_train = read_samples(neg_train_dir, preprocess=preproc_tok)
    neg_test = read_samples(neg_test_dir, preprocess=preproc_tok)
    
    
    pos = pos_train + pos_test
    neg = neg_train + neg_test
    
    # For gutenberg embeddings
    nbow_pos = extract_nbow(w2v_gut, pos)
    nbow_neg = extract_nbow(w2v_gut, neg)
    corpus, labels = create_corpus(nbow_pos, nbow_neg)
    train_corpus, test_corpus, train_labels, test_labels, = train_test_split(corpus, labels)

    classifier = train_sentiment_analysis(train_corpus, train_labels)
    predictions = evaluate_sentiment_analysis(classifier, test_corpus, test_labels)
    


    print("\n\n\n\n\n Now Google")
    print("-" * 70)
    # For google embeddings
    nbow_pos = extract_nbow(w2v_google, pos)
    nbow_neg = extract_nbow(w2v_google, neg)
    corpus, labels = create_corpus(nbow_pos, nbow_neg)
    train_corpus, test_corpus, train_labels, test_labels, = train_test_split(corpus, labels)

    classifier = train_sentiment_analysis(train_corpus, train_labels)
    predictions = evaluate_sentiment_analysis(classifier, test_corpus, test_labels)