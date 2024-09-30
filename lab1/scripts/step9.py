import subprocess
import math
from itertools import islice
from util import format_arc
import numpy as np

#9b
word_freqs = {}
with open("vocab/words.vocab.txt", 'r') as f:
    for line in f:
        word, idx = line.strip().split('\t')
        word_freqs[word] = int(idx)  # Store frequency as integer

total_words = sum(word_freqs.values())

with open("fsts/W.fst", 'w') as f:
    f.write("0\n")  # Initial (and only) state
    for word, freq in word_freqs.items():
        if word == "<eps>":
            continue  # Skip epsilon symbol
        prob = freq / total_words  # Calculate probability
        cost = -math.log10(prob)     # Negative log probability
        f.write(f"0\t0\t{word}\t{word}\t{cost:.6f}\n")  # Write transition
    f.write("0\n")  # Final state



subprocess.run([
    "fstcompile", "--isymbols=vocab/words.syms", 
    "--osymbols=vocab/words.syms", "fsts/W.fst", "fsts/W.binfst"
])

#9c, 9d

# Paths to FST files
levenshtein_fst = "fsts/L.binfst"
acceptor_fst = "fsts/V.binfst"
language_model_fst = "fsts/W.binfst"
edit_fst = "fsts/E.binfst"

# Create intermediate FSTs
EV_fst = "fsts/EV.binfst"
LV_fst = "fsts/S.binfst" #from step 6
VW_fst = "fsts/VW.binfst"
LVW_fst = "fsts/LVW.binfst"
EVW_fst = "fsts/EVW.binfst"

# Create VW
subprocess.run(["fstarcsort", "--sort_type=olabel", acceptor_fst, acceptor_fst])
subprocess.run(["fstarcsort", "--sort_type=ilabel", language_model_fst, language_model_fst])
subprocess.run(["fstcompose", acceptor_fst, language_model_fst, VW_fst])

# Create LVW
subprocess.run(["fstarcsort", "--sort_type=olabel", LV_fst, LV_fst])
subprocess.run(["fstarcsort", "--sort_type=ilabel", language_model_fst, language_model_fst])
subprocess.run(["fstcompose", LV_fst, language_model_fst, LVW_fst])

# Create EVW
subprocess.run(["fstarcsort", "--sort_type=olabel", EV_fst, EV_fst])
subprocess.run(["fstarcsort", "--sort_type=ilabel", language_model_fst, language_model_fst])
subprocess.run(["fstcompose", EV_fst, language_model_fst, EVW_fst])

print("\n")

#9e
print("LVW")
print("-"*50)
subprocess.run(["python3","scripts/step7.py", "fsts/LVW.binfst"])

#9στ
print("Results for LVW")
print("For cwt: ")
subprocess.run(["./scripts/predict.sh", "fsts/LVW.binfst", "cwt"])
print("For cit: ")
subprocess.run(["./scripts/predict.sh", "fsts/LVW.binfst", "cit"])
print()
print("Results for LV")
print("For cwt: ")
subprocess.run(["./scripts/predict.sh", "fsts/LV.binfst", "cwt"])
print()
print("For cit: ")
subprocess.run(["./scripts/predict.sh", "fsts/LV.binfst", "cit"])
print()

#9ζ
with open("fsts/W_small.fst", 'w') as f:
    f.write("0\n")  # Initial (and only) state
    num_words = 9
    for word, freq in islice(word_freqs.items(), num_words):
        if word == "<eps>":
            continue  # Skip epsilon symbol
        prob = freq / total_words  # Calculate probability
        cost = -math.log(prob)     # Negative log probability
        f.write(f"0 0 {word} {word} {cost:.6f}\n")  # Write transition
    f.write("0\n")  # Final state

subprocess.run([
    "fstcompile", "--isymbols=vocab/words.syms", 
    "--osymbols=vocab/words.syms", "fsts/W_small.fst", "fsts/W_small.binfst"
])

# Draw the subset W_small.fst and convert to PDF
subprocess.run([
    "fstdraw", "--isymbols=vocab/words.syms", "--osymbols=vocab/words.syms", 
    "fsts/W_small.binfst", "fsts/dots/W_small.dot"
])
subprocess.run(["dot", "-Tpdf", "fsts/dots/W_small.dot", "-o", "fsts/pdfs/W_small.pdf"])

# We have V_small fst from previous steps
# Create VW
subprocess.run(["fstarcsort",  "--sort_type=olabel", "fsts/V_small.binfst", "fsts/V_small.binfst"])
subprocess.run(["fstarcsort", "--sort_type=ilabel", "fsts/W_small.binfst", "fsts/W_small.binfst"])
subprocess.run(["fstrmepsilon", "fsts/W_small.binfst", "|", "fstdeterminize", "|", "fstminimize"])
subprocess.run(["fstcompose", "fsts/V_small.binfst", "fsts/W_small.binfst", "fsts/VW_small.binfst"])

# Draw the subset VW_small.fst and convert to PDF
subprocess.run([
    "fstdraw", "--isymbols=vocab/words.syms", "--osymbols=vocab/words.syms", 
    "fsts/VW_small.binfst", "fsts/dots/VW_small.dot"
])
subprocess.run(["dot", "-Tpdf", "fsts/dots/VW_small.dot", "-o", "fsts/pdfs/VW_small.pdf"])

















