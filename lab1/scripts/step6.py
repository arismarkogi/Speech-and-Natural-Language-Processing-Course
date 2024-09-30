import subprocess

def compose_transducer_acceptor(levenshtein_fst, acceptor_fst, output_fst):
    """Sorts the two transducers and Composes the Levenshtein transducer with the acceptor FST."""
    subprocess.run(["fstarcsort", "--sort_type=ilabel", acceptor_fst, acceptor_fst])
    subprocess.run(["fstarcsort", "--sort_type=olabel", levenshtein_fst, levenshtein_fst])
    subprocess.run(["fstcompose", levenshtein_fst, acceptor_fst, output_fst])


# paths to  FST files
levenshtein_fst = "fsts/L.binfst"
acceptor_fst = "fsts/V.binfst"
spell_checker_fst = "fsts/S.binfst"

# Compose the spell checker FST
compose_transducer_acceptor(levenshtein_fst, acceptor_fst, spell_checker_fst)

