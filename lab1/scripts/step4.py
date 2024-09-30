import string
from collections import Counter
import subprocess  # For executing shell commands (fstcompile, fstdraw)
import os

# Create Levenshtein transducer (Step 4)

# 4b
with open("fsts/L.fst", "w") as f:
    f.write("0 0 <eps> <eps> 0\n")
    for char in string.ascii_lowercase:
        f.write(f"0 0 {char} {char} 0\n")
        f.write(f"0 0 <eps> {char} 1\n")
        f.write(f"0 0 {char} <eps> 1\n")
        for other_char in string.ascii_lowercase:
            if char != other_char:
                f.write(f"0 0 {char} {other_char} 1\n")
    f.write("0\n")  


# Compile and draw the transducer

#4c
subprocess.run([
    "fstcompile", "--isymbols=vocab/chars.syms", 
    "--osymbols=vocab/chars.syms", "fsts/L.fst", "fsts/L.binfst"
])

#4ζ

# Filter the FST to include only a portion of it
subset_chars = ['a', 'b', 'c']  # Define a subset for easier visualization

# Write the subset FST to a file in OpenFst text format
with open("fsts/L_subset.fst", "w") as f:
    f.write("0 0 <eps> <eps> 0\n")
    for char in subset_chars:
        f.write(f"0 0 {char} {char} 0\n")
        f.write(f"0 0 <eps> {char} 1\n")
        f.write(f"0 0 {char} <eps> 1\n")
        for other_char in subset_chars:
            if char != other_char:
                f.write(f"0 0 {char} {other_char} 1\n")
    f.write("0\n")

# Compile the subset FST
subprocess.run([
    "fstcompile", "--isymbols=vocab/chars.syms",
    "--osymbols=vocab/chars.syms", "fsts/L_subset.fst", "fsts/L_subset.binfst"
])

# Draw the subset FST and convert it to PDF with additional options

subprocess.run([
    "fstdraw", "--isymbols=vocab/chars.syms", "--osymbols=vocab/chars.syms", 
    "fsts/L_subset.binfst", "fsts/dots/L_subset.dot"
])
subprocess.run(["dot", "-Tpdf", "fsts/dots/L_subset.dot", "-o", "fsts//pdfs/L_subset.pdf"])