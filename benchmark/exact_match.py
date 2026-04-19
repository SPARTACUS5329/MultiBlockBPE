# # # import subprocess
# # # import tempfile
# # # from tokenizers import Tokenizer

# # # def tokenize(input_path):
# # #     result = subprocess.run(
# # #         ["./build/MultiBlockBPE", input_path, "stdout", "BATCH", "256"],
# # #         capture_output=True,
# # #         text=True
# # #     )

# # #     # If tokenizer crashes, don't kill the whole pipeline
# # #     if result.returncode != 0:
# # #         print("❌ Tokenizer failed on file:", input_path)
# # #         print(result.stderr)
# # #         return []

# # #     return result.stdout.strip().split("\n")


# # # def chunk_file(file_path, words_per_chunk=100000):
# # #     with open(file_path, "r", encoding="utf-8") as f:
# # #         buffer = []

# # #         for line in f:
# # #             for word in line.split():
# # #                 buffer.append(word)

# # #                 if len(buffer) == words_per_chunk:
# # #                     yield " ".join(buffer)
# # #                     buffer = []

# # #         # leftover chunk
# # #         if buffer:
# # #             yield " ".join(buffer)


# # # if __name__ == "__main__":
# # #     input_path = "./assets/wap.txt"
# # #     output_path = "tokens_500_chunks.txt"

# # #     with open(output_path, "w", encoding="utf-8") as out:

# # #         for i, chunk in enumerate(chunk_file(input_path, 100000)):

# # #             # 🔍 Debug info
# # #             print(f"\nProcessing chunk {i}")
# # #             print("Word count:", len(chunk.split()))
# # #             print("Preview:", repr(chunk[:120]))

# # #             # 🚨 Safety check (prevents many segfaults)
# # #             if len(chunk.split()) < 5:
# # #                 print("Skipping tiny chunk")
# # #                 continue

# # #             # Write chunk to temp file
# # #             with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
# # #                 tmp.write(chunk)
# # #                 tmp.flush()

# # #                 tokens = tokenize(tmp.name)

# # #             # Save output
# # #             out.write(f"--- Chunk {i} ---\n")
# # #             out.write(" ".join(tokens) + "\n\n")

# # #     print(f"\n✅ Saved tokenized output to {output_path}")




# import subprocess
# import tempfile
# from tokenizers import Tokenizer

# INPUT_FILE = "./assets/wap.txt"
# SUMMARY_LOG = "comparison_summary.txt"
# HF_TOKENS_LOG = "hf_tokens_output.txt"
# CPP_TOKENS_LOG = "cpp_tokens_output.txt"

# CONFIGS = [
#     # (256, 128),
#     # (256, 256),
#     # ... uncomment your other configs as needed ...
#     (256, 128)
# ]

# def tokenize_cpp(input_path, bs):
#     """Runs the C++ tokenizer and returns the token output as a list of strings."""
#     result = subprocess.run(
#         ["./build/MultiBlockBPE", input_path, "stdout", "BATCH", str(bs)],
#         capture_output=True,
#         text=True
#     )
#     if result.returncode != 0:
#         print("Tokenizer failed on file:", input_path)
#         print(result.stderr)
#         return []
    
#     return result.stdout.strip().split()

# def chunk_file(file_path, words_per_chunk=100000):
#     """Yields chunks of text containing exactly words_per_chunk words."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         buffer = []
#         for line in f:
#             for word in line.split():
#                 buffer.append(word)
#                 if len(buffer) == words_per_chunk:
#                     yield " ".join(buffer)
#                     buffer = []
#         # leftover chunk
#         if buffer:
#             yield " ".join(buffer)

# def main():
#     print("Loading HF tokenizer...")
#     hf_tok = Tokenizer.from_pretrained("gpt2")
    
#     print("Chunking text into 100,000 word segments...")
#     chunks = []
#     for i, chunk in enumerate(chunk_file(INPUT_FILE, 100000)):
#         if len(chunk.split()) < 5:
#             print(f"Skipping tiny chunk {i}")
#             continue
#         chunks.append(chunk)
    
#     print(f"Total chunks created: {len(chunks)}\n")

#     # Open all output files
#     with open(SUMMARY_LOG, "w", encoding="utf-8") as summary_out, \
#          open(HF_TOKENS_LOG, "w", encoding="utf-8") as hf_out, \
#          open(CPP_TOKENS_LOG, "w", encoding="utf-8") as cpp_out:

#         for bs, seqlen in CONFIGS:
#             print(f"--- Running Config: Batch Size {bs}, Seq Length {seqlen} ---")
            
#             is_1_to_1 = True

#             for i, chunk in enumerate(chunks):
#                 # 1. Tokenize with HuggingFace
#                 hf_encoded = hf_tok.encode(chunk)
#                 hf_token_ids = [str(tid) for tid in hf_encoded.ids]

#                 # 2. Tokenize with C++ MultiBlockBPE
#                 with tempfile.NamedTemporaryFile(mode="w+", delete=True, encoding="utf-8") as tmp:
#                     tmp.write(chunk)
#                     tmp.flush()
#                     cpp_token_ids = tokenize_cpp(tmp.name, bs)

#                 # 3. Store both outputs for every chunk
#                 header = f"--- Config BS:{bs} SEQ:{seqlen} Chunk {i} ---\n"
#                 hf_out.write(header + " ".join(hf_token_ids) + "\n\n")
#                 cpp_out.write(header + " ".join(cpp_token_ids) + "\n\n")

#                 # 4. Compare tokens
#                 if hf_token_ids != cpp_token_ids:
#                     is_1_to_1 = False
#                     print(f"Mismatch found in chunk {i}!")
#                     print(f"   HF Token Count : {len(hf_token_ids)}")
#                     print(f"   C++ Token Count: {len(cpp_token_ids)}")
                    
#                     min_len = min(len(hf_token_ids), len(cpp_token_ids))
#                     for idx in range(min_len):
#                         if hf_token_ids[idx] != cpp_token_ids[idx]:
#                             print(f"   -> Diff at index {idx}: HF='{hf_token_ids[idx]}', C++='{cpp_token_ids[idx]}'")
#                             break
                    
#                     if len(hf_token_ids) != len(cpp_token_ids):
#                         print("   -> (Lengths differ)")

#             # Output results for this config to summary
#             if is_1_to_1:
#                 print("Splits match 1:1 perfectly with C++ output.")
#             else:
#                 print("Splits DO NOT match 1:1.")

#             summary_out.write(f"Batch size: {bs}\n")
#             summary_out.write(f"Seq length: {seqlen}\n")
#             summary_out.write(f"1:1 ID Match with C++: {is_1_to_1}\n\n")
            
#             print(f"Done: batch={bs}, seq={seqlen}\n")

#     print("Benchmarking complete.")
#     print(f"Summary written to {SUMMARY_LOG}")
#     print(f"HuggingFace tokens written to {HF_TOKENS_LOG}")
#     print(f"C++ tokens written to {CPP_TOKENS_LOG}")

# if __name__ == "__main__":
#     main()


import subprocess
import tempfile
from tokenizers import Tokenizer

INPUT_FILE = "./assets/wap.txt"
SUMMARY_LOG = "comparison_summary.txt"
HF_TOKENS_LOG = "hf_tokens_output.txt"
CPP_TOKENS_LOG = "cpp_tokens_output.txt"

CONFIGS = [
    # (256, 128),
    # (256, 256),
    # ... uncomment your other configs as needed ...
    (256, 128)
]

def tokenize_cpp(input_path, bs):
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
        if len(chunk.split()) < 5:
            print(f"Skipping tiny chunk {i}")
            continue
        chunks.append(chunk)
    
    print(f"Total chunks created: {len(chunks)}\n")

    # Open all output files
    with open(SUMMARY_LOG, "w", encoding="utf-8") as summary_out, \
         open(HF_TOKENS_LOG, "w", encoding="utf-8") as hf_out, \
         open(CPP_TOKENS_LOG, "w", encoding="utf-8") as cpp_out:

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
                    cpp_token_ids = tokenize_cpp(tmp.name, bs)

                # 3. Store both outputs for every chunk (with heavy visual separation)
                divider = "=" * 60
                header = f"{divider}\n--- Config BS:{bs} SEQ:{seqlen} | CHUNK {i} ---\n{divider}\n"
                
                hf_out.write(header + " ".join(hf_token_ids) + "\n\n\n\n\n")
                cpp_out.write(header + " ".join(cpp_token_ids) + "\n\n\n\n\n")

                # 4. Compare tokens
                if hf_token_ids != cpp_token_ids:
                    is_1_to_1 = False
                    print(f"Mismatch found in chunk {i}!")
                    print(f"   HF Token Count : {len(hf_token_ids)}")
                    print(f"   C++ Token Count: {len(cpp_token_ids)}")
                    
                    min_len = min(len(hf_token_ids), len(cpp_token_ids))
                    for idx in range(min_len):
                        if hf_token_ids[idx] != cpp_token_ids[idx]:
                            print(f"   -> Diff at index {idx}: HF='{hf_token_ids[idx]}', C++='{cpp_token_ids[idx]}'")
                            break
                    
                    if len(hf_token_ids) != len(cpp_token_ids):
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