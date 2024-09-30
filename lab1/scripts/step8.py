import subprocess
import string
import math 


#8c
word_pairs = [("tst", "test", "insertion"), ("applle", "apple", "deletion"), ("workong", "working", "substitution")]
for wrong, correct, edit in word_pairs:
    print(f"Wrong word: {wrong}, Correct: {correct}, Edit to check: {edit}")
    subprocess.run(["./scripts/word_edits.sh", wrong, correct])


#8d 
with open("data/wiki.txt", 'r') as file_read:
    with open('data/wiki_edits.txt', 'w') as file_write:
        for line_read in file_read:
            
            wrong, correct = line_read.strip().split('\t')
            result = subprocess.run(
                ["scripts/word_edits.sh", wrong, correct], 
                capture_output=True,  
                text=True
            )
            file_write.write(result.stdout)  

#8e

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
    
    
#8στ

# Create E.fst with log probabilities
with open("fsts/E.fst", "w") as f:
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

# Compile the E.fst
subprocess.run([
    "fstcompile", "--isymbols=vocab/chars.syms", 
    "--osymbols=vocab/chars.syms", "fsts/E.fst", "fsts/E.binfst"
])

# Draw E.fst for a subset

# Filter to subset of characters
subset_chars = ['a', 'b', 'c']

# Write the subset E.fst to a file
with open("fsts/E_subset.fst", "w") as f:
    for char in subset_chars:
        f.write(f"0 0 {char} {char} 0\n")  # No cost for correct characters

        # Insertions and deletions
        insert_cost = -math.log(edits_dict.get(("<eps>", char), 1e-10))
        delete_cost = -math.log(edits_dict.get((char, "<eps>"), 1e-10))
        f.write(f"0 0 <eps> {char} {insert_cost:.6f}\n") 
        f.write(f"0 0 {char} <eps> {delete_cost:.6f}\n")

        # Substitutions
        for other_char in subset_chars:
            if char != other_char:
                sub_cost = -math.log(edits_dict.get((char, other_char), 1e-10))
                f.write(f"0 0 {char} {other_char} {sub_cost:.6f}\n")

    f.write("0\n")  # Final state

# Compile the subset E.fst
subprocess.run([
    "fstcompile", "--isymbols=vocab/chars.syms",
    "--osymbols=vocab/chars.syms", "fsts/E_subset.fst", "fsts/E_subset.binfst"
])

# Draw the subset E.fst and convert to PDF
subprocess.run([
    "fstdraw", "--isymbols=vocab/chars.syms", "--osymbols=vocab/chars.syms", 
    "fsts/E_subset.binfst", "fsts/dots/E_subset.dot"
])
subprocess.run(["dot", "-Tpdf", "fsts/dots/E_subset.dot", "-o", "fsts/pdfs/E_subset.pdf"]) 





#8ζ


def compose_transducer_acceptor(E_fst_fst, acceptor_fst, output_fst):
    """Sorts the two transducers and Composes the E_fst transducer with the acceptor FST."""
    subprocess.run(["fstarcsort", "--sort_type=ilabel", acceptor_fst, acceptor_fst])
    subprocess.run(["fstarcsort", "--sort_type=olabel", E_fst_fst, E_fst_fst])
    subprocess.run(["fstcompose", E_fst_fst, acceptor_fst, output_fst])


# paths to  FST files
E_fst = "fsts/E.binfst"
acceptor_fst = "fsts/V.binfst"
EV_fst = "fsts/EV.binfst"

# Compose the spell checker FST
compose_transducer_acceptor(E_fst, acceptor_fst, EV_fst)

subprocess.run(["python3","scripts/step7.py", "fsts/EV.binfst"])



