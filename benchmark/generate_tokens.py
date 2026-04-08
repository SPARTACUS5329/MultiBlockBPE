import subprocess

RAW_DATASET = "./data/mmlu_raw.txt"
TOKENIZER_BINARY = "./build/MultiBlockBPE"
TOKEN_OUTPUT_FILE = "./data/mmlu_token_ids.txt"
MODE = "BATCH"
VALUE = "256"

print("Running MultiBlockBPE tokenizer...")

subprocess.run(
    [TOKENIZER_BINARY, RAW_DATASET, TOKEN_OUTPUT_FILE, MODE, VALUE],
    check=True
)