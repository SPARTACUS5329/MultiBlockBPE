from rapidfuzz.distance import Levenshtein

# Load sequences
hf_ids = list(map(int, open("hf_tokens3.txt").read().split()))
bbpe_ids = list(map(int, open("blockbpe_tokens3.txt").read().split()))

# Levenshtein on sequences of integers
dist = Levenshtein.distance(hf_ids, bbpe_ids)

similarity = 1 - dist / max(len(hf_ids), len(bbpe_ids))

print("Token-level Levenshtein distance:", dist)
print("Similarity:", similarity)
