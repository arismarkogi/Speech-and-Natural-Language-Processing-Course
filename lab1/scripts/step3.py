import string
import re
import sys

# a) Create chars.syms
with open("vocab/chars.syms", "w") as f:
    f.write("<eps>\t0\n")  # Epsilon symbol
    for i, char in enumerate(string.ascii_lowercase):
        f.write(f"{char}\t{i+1}\n")  # Assign indices starting from 1

# b) Create words.syms
with open("vocab/words.vocab.txt", "r") as f_vocab, open("vocab/words.syms", "w") as f_syms:
    f_syms.write("<eps>\t0\n")  
    for i, line in enumerate(f_vocab):
        word, _ = line.strip().split("\t")  # Extract word (ignore count)
        f_syms.write(f"{word}\t{i+1}\n")