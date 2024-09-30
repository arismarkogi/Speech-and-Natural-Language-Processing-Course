import subprocess
import string
import math


# 11a
edits_dict = {}
with open("data/wiki_edits.txt", 'r') as file_read:
    for line_read in file_read:
        arg1, arg2 = line_read.strip().split('\t')
        if (arg1, arg2) in edits_dict:
            edits_dict[(arg1, arg2)] += 1
        else:
            edits_dict[(arg1, arg2)] = 1

total_edits = sum(edits_dict.values())

# Normalize edit counts to frequencies
for edit, count in edits_dict.items():
    edits_dict[edit] = count / total_edits
    
# Add-1 Smoothing
V = len(edits_dict)  # Number of unique observed edits
unseen_edit_prob = 1 / (total_edits + V)  # Probability mass for unseen edits

for char in string.ascii_lowercase:
    for other_char in string.ascii_lowercase:
        edit = (char, other_char)
        if edit not in edits_dict:
            edits_dict[edit] = unseen_edit_prob

# Normalize edit counts (now including smoothed probabilities)
for edit, count in edits_dict.items():
    edits_dict[edit] = count / (total_edits + V)  # Normalize with smoothed total

# Create E_new.fst with log probabilities
with open("fsts/E_new.fst", "w") as f:
    for char in string.ascii_lowercase:
        f.write(f"0 0 {char} {char} 0\n")  # No cost for correct characters

        # Insertions and deletions
        insert_cost = -math.log(edits_dict.get(("<eps>", char), 1e-10))  # 1e-10 for unseen edits
        delete_cost = -math.log(edits_dict.get((char, "<eps>"), 1e-10))
        f.write(f"0 0 <eps> {char} {insert_cost:.6f}\n")  # Limit to 6 decimal places
        f.write(f"0 0 {char} <eps> {delete_cost:.6f}\n")

        # Substitutions
        for other_char in string.ascii_lowercase:
            if char != other_char:
                sub_cost = -math.log(edits_dict.get((char, other_char), 1e-10))
                f.write(f"0 0 {char} {other_char} {sub_cost:.6f}\n")

    f.write("0\n")  # Final state

# Compile the E_new.fst
subprocess.run([
    "fstcompile", "--isymbols=vocab/chars.syms", 
    "--osymbols=vocab/chars.syms", "fsts/E_new.fst", "fsts/E_new.binfst"
])

def compose_transducer_acceptor(E_fst_fst, acceptor_fst, output_fst):
    
    subprocess.run(["fstarcsort", "--sort_type=ilabel", acceptor_fst, acceptor_fst])
    subprocess.run(["fstarcsort", "--sort_type=olabel", E_fst_fst, E_fst_fst])
    subprocess.run(["fstcompose", E_fst_fst, acceptor_fst, output_fst])


# paths to  FST files
E_fst = "fsts/E_new.binfst"
acceptor_fst = "fsts/V.binfst"
EV_fst = "fsts/EV_new.binfst"

# Compose the spell checker FST
compose_transducer_acceptor(E_fst, acceptor_fst, EV_fst)

subprocess.run(["python3","scripts/run_evaluation.py", "fsts/EV_new.binfst"])


# 11b

from collections import Counter

import re
from collections import Counter

def merge_word_counts(file1_path, file2_path):
    pattern = re.compile(r'[^a-zA-Z\s]')  # Match non-English letters/whitespace

    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        counts1 = Counter({word: int(count) for word, count in (line.strip().split('\t') for line in f1)})
        counts2 = Counter({word: int(count) for word, count in (line.strip().split() for line in f2)})

    filtered_counts1 = Counter({word: count for word, count in counts1.items() if not pattern.search(word)})
    filtered_counts2 = Counter({word: count for word, count in counts2.items() if not pattern.search(word)})

    merged_counts = filtered_counts1 + filtered_counts2  # Combine frequencies
    return merged_counts


# Get file paths from the user or define them here
file1_path = 'vocab/words.vocab.txt'
file2_path = 'data/en_50k.txt'

merged_counts = merge_word_counts(file1_path, file2_path)

# Write to a new file
with open('vocab/merged_vocab.txt', 'w') as f:
    for word, count in merged_counts.items():
        f.write(f"{word}\t{count}\n")



# Create new_words.syms
with open("vocab/merged_vocab.txt", "r") as f_vocab, open("vocab/new_words.syms", "w") as f_syms:
    f_syms.write("<eps>\t0\n")  
    for i, line in enumerate(f_vocab):
        word, _ = line.strip().split("\t")  # Extract word (ignore count)
        f_syms.write(f"{word}\t{i+1}\n")

#5a
with open("vocab/merged_vocab.txt", "r") as f_vocab, open("fsts/V_new.fst", "w") as f_fst:
    state_counter = 2  # Start from 2, as 1 is implicitly initial
    initial_state = 1 

    f_fst.write(f"{initial_state}\n")  # Explicit initial state declaration

    for line in f_vocab:
        word, _ = line.strip().split("\t")

        current_state = initial_state  # Each word begins at initial state

        for char in word:
            next_state = state_counter
            f_fst.write(f"{current_state} {next_state} {char} {char} 0\n")
            current_state = next_state
            state_counter += 1

        # Epsilon transition back to initial state and final weight
        f_fst.write(f"{current_state} 0 <eps> <eps> 0\n") 
    f_fst.write("0\n")


subprocess.run([
        "fstcompile",
        "--isymbols=vocab/chars.syms",
        "--osymbols=vocab/new_words.syms",
        "fsts/V_new.fst",
        "fsts/V_new.binfst",
    ])

print("finished compile")

# 5b, #5c
subprocess.run(["fstrmepsilon", "fsts/V_new.binfst", "fsts/V_new.binfst"])

print("finished fstrmepsilon")
subprocess.run(["fstdeterminize", "fsts/V_new.binfst", "fsts/V_new.binfst"])

print("finished fstdeterminize")
subprocess.run(["fstminimize", "fsts/V_new.binfst", "fsts/V_new.binfst"])



word_freqs = {}
with open("vocab/merged_vocab.txt", 'r') as f:
    for line in f:
        word, idx = line.strip().split()
        word_freqs[word] = int(idx)  # Store frequency as integer

total_words = sum(word_freqs.values())

with open("fsts/W_new.fst", 'w') as f:
    f.write("0\n")  # Initial (and only) state
    for word, freq in word_freqs.items():
        if word == "<eps>":
            continue  # Skip epsilon symbol
        prob = freq / total_words  # Calculate probability
        cost = -math.log10(prob)     # Negative log probability
        f.write(f"0\t0\t{word}\t{word}\t{cost:.6f}\n")  # Write transition
    f.write("0\n")  # Final state



subprocess.run([
    "fstcompile", "--isymbols=vocab/new_words.syms", 
    "--osymbols=vocab/new_words.syms", "fsts/W_new.fst", "fsts/W_new.binfst"
])

# Create E_new_V_new
subprocess.run(["fstarcsort", "--sort_type=olabel", "fsts/E_new.binfst", "fsts/E_new.binfst"])
subprocess.run(["fstarcsort", "--sort_type=ilabel", "fsts/V_new.binfst", "fsts/V_new.binfst"])
subprocess.run(["fstcompose", "fsts/E_new.binfst", "fsts/V_new.binfst", "fsts/E_new_V_new.binfst"])


# Create EVW_new
subprocess.run(["fstarcsort", "--sort_type=olabel", "fsts/E_new_V_new.binfst", "fsts/E_new_V_new.binfst"])
subprocess.run(["fstarcsort", "--sort_type=ilabel", "fsts/W_new.binfst", "fsts/W_new.binfst"])
subprocess.run(["fstcompose", "fsts/E_new_V_new.binfst", "fsts/W_new.binfst", "fsts/EVW_new.binfst"])

subprocess.run(["python3","scripts/run_evaluation.py", "fsts/EVW_new.binfst"])
