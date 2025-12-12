from Levenshtein import distance

# Load decoded texts
with open("hf_decoded.txt", "r", encoding="utf8") as f:
    hf_text = f.read()

with open("blockbpe_decoded.txt", "r", encoding="utf8") as f:
    bpe_text = f.read()

# OPTIONAL: remove whitespace differences
hf_text_clean  = "".join(hf_text.split())
bpe_text_clean = "".join(bpe_text.split())

# Compute character-level edit distance
dist = distance(hf_text_clean, bpe_text_clean)

# Normalize by length
max_len = max(len(hf_text_clean), len(bpe_text_clean))
sim = 1 - dist / max_len if max_len > 0 else 1.0

print(f"Levenshtein distance: {dist}")
print(f"Similarity: {sim:.6f}")
