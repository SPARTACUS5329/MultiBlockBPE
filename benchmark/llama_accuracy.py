import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

# -----------------------------
# Argument Parsing
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--hf_dir", type=str, required=True)
parser.add_argument("--mb_dir", type=str, required=True)
parser.add_argument("--gold", type=str, required=True)
parser.add_argument("--task", type=str, choices=["mc", "numeric"], required=True)
parser.add_argument("--max_new_tokens", type=int, default=128)

args = parser.parse_args()

# -----------------------------
# Load Model
# -----------------------------

model_name = "meta-llama/Llama-3.1-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

vocab_size = model.config.vocab_size

# -----------------------------
# Utility Functions
# -----------------------------

def load_token_ids(filepath):
    with open(filepath, "r") as f:
        tokens = f.read().strip().split()
    ids = torch.tensor([int(x) for x in tokens], dtype=torch.long)

    if ids.max() >= vocab_size:
        raise ValueError(f"Token ID exceeds vocab size in {filepath}")

    # Ensure BOS token
    if tokenizer.bos_token_id is not None:
        if ids[0].item() != tokenizer.bos_token_id:
            ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), ids])

    return ids


def run_inference(token_ids):
    input_ids = token_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return output_ids[0]


def extract_mc_answer(output_ids):
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Strict letter extraction
    for choice in ["A", "B", "C", "D"]:
        if re.search(rf"\b{choice}\b", text):
            return choice

    return None


def extract_numeric_answer(output_ids):
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    matches = re.findall(r"-?\d+\.?\d*", text)
    return matches[-1] if matches else None


def evaluate(token_dir, gold_answers):
    correct = 0
    total = 0

    files = sorted(os.listdir(token_dir))

    for filename in tqdm(files):
        sample_id = os.path.splitext(filename)[0]
        gold = gold_answers.get(sample_id)

        if gold is None:
            continue

        token_path = os.path.join(token_dir, filename)
        token_ids = load_token_ids(token_path)
        output_ids = run_inference(token_ids)

        if args.task == "mc":
            pred = extract_mc_answer(output_ids)
        else:
            pred = extract_numeric_answer(output_ids)

        if pred is not None and str(pred).strip() == str(gold).strip():
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0


# -----------------------------
# Main Evaluation
# -----------------------------

print("Loading gold answers...")
with open(args.gold, "r") as f:
    gold_answers = json.load(f)

print("\nEvaluating HF tokenization...")
hf_acc = evaluate(args.hf_dir, gold_answers)

print("\nEvaluating MultiBlockBPE tokenization...")
mb_acc = evaluate(args.mb_dir, gold_answers)

print("\n----------------------------")
print(f"HF Accuracy:        {hf_acc:.4f}")
print(f"MultiBlock Accuracy:{mb_acc:.4f}")
print(f"Delta:              {mb_acc - hf_acc:.4f}")
print("----------------------------")
