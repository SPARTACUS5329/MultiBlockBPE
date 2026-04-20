import subprocess
import tempfile
from tokenizers import Tokenizer

INPUT_FILE = "./assets/wap.txt"
SUMMARY_LOG = "comparison_summary.txt"
HF_TOKENS_LOG = "hf_tokens_output.txt"
CPP_TOKENS_LOG = "multiblockbpe_tokens_output.txt"

CONFIGS = [
    (256, 128) # (BATCH_SIZE, SEQ_LEN)
]

def multiblock_bpe(input_path, bs):
    """Runs the C++ tokenizer and returns the token output as a list of strings."""
    result = subprocess.run(
        ["./build/MultiBlockBPE", input_path, "stdout", "BATCH", str(bs)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Tokenizer failed on file:", input_path)
        print(result.stderr)
        return []
    
    return result.stdout.strip().split()

def chunk_file(file_path, words_per_chunk=50000):
    """Yields chunks of text containing exactly words_per_chunk words."""
    with open(file_path, "r", encoding="utf-8") as f:
        buffer = []
        for line in f:
            for word in line.split():
                buffer.append(word)
                if len(buffer) == words_per_chunk:
                    yield " ".join(buffer)
                    buffer = []
        # leftover chunk
        if buffer:
            yield " ".join(buffer)

def main():
    print("Loading HF tokenizer...")
    hf_tok = Tokenizer.from_pretrained("gpt2")
    
    print("Chunking text into 100,000 word segments...")
    chunks = []
    for i, chunk in enumerate(chunk_file(INPUT_FILE, 50000)):
        # if len(chunk.split()) < 5:
        #     print(f"Skipping tiny chunk {i}")
        #     continue
        chunks.append(chunk)
    
    print(f"Total chunks created: {len(chunks)}\n")

    # Open all output files
    with open(SUMMARY_LOG, "w", encoding="utf-8") as summary_out, \
         open(HF_TOKENS_LOG, "w", encoding="utf-8") as hf_out, \
         open(CPP_TOKENS_LOG, "w", encoding="utf-8") as multiblock_bpe_out:

        for bs, seqlen in CONFIGS:
            print(f"--- Running Config: Batch Size {bs}, Seq Length {seqlen} ---")
            
            is_1_to_1 = True

            for i, chunk in enumerate(chunks):
                # 1. Tokenize with HuggingFace
                hf_encoded = hf_tok.encode(chunk)
                hf_token_ids = [str(tid) for tid in hf_encoded.ids]

                # 2. Tokenize with C++ MultiBlockBPE
                with tempfile.NamedTemporaryFile(mode="w+", delete=True, encoding="utf-8") as tmp:
                    tmp.write(chunk)
                    tmp.flush()
                    multiblock_bpe_ids = multiblock_bpe(tmp.name, bs)
                    merges_str_id = multiblock_bpe_ids.index("merges:")
                    strip_lb = merges_str_id + 1
                    batch_str_id = multiblock_bpe_ids.index("Batch")
                    strip_ub = batch_str_id
                    multiblock_bpe_ids = multiblock_bpe_ids[strip_lb:strip_ub]
                    multiblock_bpe_ids.pop(0) # There is a random "frog" token as the first one in each chunk

                # 3. Store both outputs for every chunk (with heavy visual separation)
                divider = "=" * 60
                header = f"{divider}\n--- Config BS:{bs} SEQ:{seqlen} | CHUNK {i} ---\n{divider}\n"
                
                hf_out.write(header + " ".join(hf_token_ids) + "\n\n\n\n\n")
                multiblock_bpe_out.write(header + " ".join(multiblock_bpe_ids) + "\n\n\n\n\n")

                # 4. Compare tokens
                if hf_token_ids != multiblock_bpe_ids:
                    is_1_to_1 = False
                    print(f"Mismatch found in chunk {i}!")
                    print(f"   HF Token Count : {len(hf_token_ids)}")
                    print(f"   C++ Token Count: {len(multiblock_bpe_ids)}")
                    
                    min_len = min(len(hf_token_ids), len(multiblock_bpe_ids))
                    for idx in range(min_len):
                        if hf_token_ids[idx] != multiblock_bpe_ids[idx]:
                            print(f"   -> Diff at index {idx}: HF='{hf_token_ids[idx]}', C++='{multiblock_bpe_ids[idx]}'")
                            break
                    
                    if len(hf_token_ids) != len(multiblock_bpe_ids):
                        print("   -> (Lengths differ)")

            # Output results for this config to summary
            if is_1_to_1:
                print("Splits match 1:1 perfectly with C++ output.")
            else:
                print("Splits DO NOT match 1:1.")

            summary_out.write(f"Batch size: {bs}\n")
            summary_out.write(f"Seq length: {seqlen}\n")
            summary_out.write(f"1:1 ID Match with C++: {is_1_to_1}\n\n")
            
            print(f"Done: batch={bs}, seq={seqlen}\n")

    print("Benchmarking complete.")
    print(f"Summary written to {SUMMARY_LOG}")
    print(f"HuggingFace tokens written to {HF_TOKENS_LOG}")
    print(f"C++ tokens written to {CPP_TOKENS_LOG}")

if __name__ == "__main__":
    main()