from gensim.models import Word2Vec

# Load the Word2Vec model
w2v = Word2Vec.load("gutenberg_w2v.100d.model").wv

# Read the vocabulary words
with open('vocab/words.syms', 'r') as file:
    words = [line.split()[0] for line in file.readlines()]

# Prepare the embeddings and metadata
embeddings = []
metadata = []

not_in_vocab = []

for word in words:
    try:
        embedding = w2v[word]
        embeddings.append('\t'.join(map(str, embedding)))
        metadata.append(word)
    except KeyError:
        not_in_vocab.append(word)

# Write embeddings to file
with open('data/embeddings.tsv', 'w') as file:
    file.write('\n'.join(embeddings) + '\n')

# Write metadata to file
with open('data/metadata.tsv', 'w') as file:
    file.write('\n'.join(metadata) + '\n')
