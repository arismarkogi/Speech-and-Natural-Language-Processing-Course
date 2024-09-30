import subprocess


# Step 5: Construct and optimize acceptor V

#5a
with open("vocab/words.vocab.txt", "r") as f_vocab, open("fsts/V.fst", "w") as f_fst:
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
        "--osymbols=vocab/words.syms",
        "fsts/V.fst",
        "fsts/V.binfst",
    ])

print("finished compile")

# 5b, #5c
subprocess.run(["fstrmepsilon", "fsts/V.binfst", "fsts/V.binfst"])

print("finished fstrmepsilon")
subprocess.run(["fstdeterminize", "fsts/V.binfst", "fsts/V.binfst"])

print("finished fstdeterminize")
subprocess.run(["fstminimize", "fsts/V.binfst", "fsts/V.binfst"])


# 5e Draw the acceptor V at different stages

subprocess.run([
        "fstcompile",
        "--isymbols=vocab/chars.syms",
        "--osymbols=vocab/small_words.syms",
        "fsts/V_small.fst",
        "fsts/V_small.binfst",
])

# Before everything
subprocess.run(
        [
            "fstdraw",
            "--isymbols=vocab/chars.syms",
            "--osymbols=vocab/small_words.syms",
            "fsts/V_small.binfst",
            "fsts/dots/V_small.dot",
        ]
    )
subprocess.run(["dot", "-Tpdf", "fsts/dots/V_small.dot", "-o", "fsts/pdfs/V_init.pdf"])



subprocess.run(["fstrmepsilon", "fsts/V_small.binfst", "fsts/V_small.binfst"])

# After fstrmepsilon
subprocess.run(
        [
            "fstdraw",
            "--isymbols=vocab/chars.syms",
            "--osymbols=vocab/small_words.syms",
            "fsts/V_small.binfst",
            "fsts/dots/V_noeps.dot",
        ]
    )
subprocess.run(["dot", "-Tpdf", "fsts/dots/V_noeps.dot", "-o", "fsts/pdfs/V_noeps.pdf"])

subprocess.run(["fstdeterminize", "fsts/V_small.binfst", "fsts/V_small.binfst"])
# After fstdeterminize

subprocess.run(
        [
            "fstdraw",
            "--isymbols=vocab/chars.syms",
            "--osymbols=vocab/small_words.syms",
            "fsts/V_small.binfst",
            "fsts/dots/V_det.dot",
        ]
    )
subprocess.run(["dot", "-Tpdf", "fsts/dots/V_det.dot", "-o", "fsts//pdfs/V_det.pdf"])


subprocess.run(["fstminimize", "fsts/V_small.binfst", "fsts/V_small.binfst"])
# After fstminimize
subprocess.run(
        [
            "fstdraw",
            "--isymbols=vocab/chars.syms",
            "--osymbols=vocab/small_words.syms",
            "fsts/V_small.binfst",
            "fsts/dots/V_min.dot",
        ]
    )
subprocess.run(["dot", "-Tpdf", "fsts/dots/V_min.dot", "-o", "fsts/pdfs/V_min.pdf"])
