# Use Python3.9
# Make sure to use the latest build

import subprocess

def tokenize(input_path):
    result = subprocess.run(
        ["./build/MultiBlockBPE", input_path, "stdout", "SEQ", "1024"],
        capture_output=True, text=True, check=True
    )
    
    tokens = result.stdout.strip().split("\n")
    return tokens

if __name__ == "__main__":
    input_path = "./assets/wap.txt"
    tokens = tokenize(input_path)
    print(tokens)