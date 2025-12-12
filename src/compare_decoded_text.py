from transformers import GPT2TokenizerFast

# Load GPT-2 tokenizer
hf_tok = GPT2TokenizerFast.from_pretrained("gpt2")

# Load HF token IDs
with open("hf_tokens_wap.txt") as f:
    hf_ids = list(map(int, f.read().split()))

# Load BlockBPE token IDs
with open("out.txt") as f:
    bbpe_ids = list(map(int, f.read().split()))

# Decode both sequences
hf_text  = hf_tok.decode(hf_ids)
bpe_text = hf_tok.decode(bbpe_ids)

# ----------------------------------------------------------
# Save decoded texts to files
# ----------------------------------------------------------
with open("hf_decoded.txt", "w", encoding="utf8") as f:
    f.write(hf_text)

with open("blockbpe_decoded.txt", "w", encoding="utf8") as f:
    f.write(bpe_text)

print("Saved decoded texts to hf_decoded.txt and blockbpe_decoded.txt")