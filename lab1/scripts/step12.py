from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Φόρτωση του εκπαιδευμένου μοντέλου (αν δεν είναι ήδη στη μνήμη)
model = Word2Vec.load("gutenberg_w2v.100d.model")

# Λήψη των word vectors από το μοντέλο
word_vectors = model.wv



# Βήμα 12γ)

# Λέξεις που θα εξετάσουμε
target_words = ["bible", "book", "bank", "water"]

# Βρίσκουμε τις πιο κοντινές λέξεις για κάθε λέξη-στόχο
for word in target_words:
    if word in word_vectors.key_to_index:
        most_similar = word_vectors.most_similar(word, topn=10)  # Παίρνουμε τις 10 πιο κοντινές
        print(f"Οι πιο κοντινές λέξεις για '{word}':")
        for similar_word, similarity in most_similar:
            print(f"  - {similar_word}: {similarity:.4f}")
    else:
        print(f"Η λέξη '{word}' δεν βρέθηκε στο λεξικό του μοντέλου.")


# Χρήση cosine similarity για σύγκριση μεταξύ δύο λέξεων
def get_cosine_similarity(word1, word2, word_vectors):
    if word1 in word_vectors.key_to_index and word2 in word_vectors.key_to_index:
        vector1 = word_vectors[word1]
        vector2 = word_vectors[word2]
        similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
        return similarity
    else:
        return None

def find_analogy_word(triplet, word_vectors):
    """Βρίσκει τη λέξη που ολοκληρώνει την αναλογία και εκτυπώνει τις 5 κορυφαίες επιλογές."""
    word_a, word_b, word_c = triplet
    if all(word in word_vectors.key_to_index for word in triplet):
        result_vector = word_vectors[word_b] - word_vectors[word_a] + word_vectors[word_c]
        most_similar = word_vectors.most_similar(positive=[result_vector], topn=5)  # Get top 5
        print(f"Top 5 similar words for the analogy '{triplet[0]} : {triplet[1]} :: {triplet[2]} :':")
        for word, similarity in most_similar:
            print(f"  - {word}: {similarity:.4f}")
        return most_similar[0][0]  # Επιστρέφουμε την κορυφαία λέξη
    else:
        print(f"Δεν βρέθηκαν όλες οι λέξεις της αναλογίας '{triplet}' στο λεξικό.")
        return None
    

# Βήμα 12δ)

# Τριπλέτες λέξεων για αναλογίες
triplets = [
    ("queen", "girl", "king"),
    ("tall", "taller", "good"),
    ("paris", "france", "london"),
]

for triplet in triplets:
    analogy_words = find_analogy_word(triplet, word_vectors)
    if not analogy_words:
        print(f"Δεν βρέθηκαν όλες οι λέξεις της αναλογίας '{triplet}' στο λεξικό.")



# Βήμα 12ε)

from gensim.models.keyedvectors import KeyedVectors

NUM_W2V_TO_LOAD = 10000000

google_news_vectors = KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True, limit=NUM_W2V_TO_LOAD
)

# Bήμα 12στ)
# Βρίσκουμε τις πιο κοντινές λέξεις για κάθε λέξη-στόχο
for word in target_words:
    if word in google_news_vectors.key_to_index:
        most_similar = google_news_vectors.most_similar(word, topn=10)  # Παίρνουμε τις 10 πιο κοντινές
        print(f"Οι πιο κοντινές λέξεις για '{word}':")
        for similar_word, similarity in most_similar:
            print(f"  - {similar_word}: {similarity:.4f}")
    else:
        print(f"Η λέξη '{word}' δεν βρέθηκε στο λεξικό του μοντέλου.")

#Βήμα 12ζ)
for triplet in triplets:
    analogy_words = find_analogy_word(triplet, google_news_vectors)
    if not analogy_words:
        print(f"Δεν βρέθηκαν όλες οι λέξεις της αναλογίας '{triplet}' στο λεξικό.")