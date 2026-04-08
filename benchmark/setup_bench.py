import os
import json
from datasets import load_dataset

DATASET_NAME = "cais/mmlu"
SPLIT = "test"
SUBJECT = "all"
MAX_SAMPLES = None  # Set to int to limit, else None

RAW_DATASET = "data/mmlu_raw.txt"
GOLD_FILE = "gold/mmlu_gold.json"

def format_prompt(sample):
    question = sample["question"]
    choices = sample["choices"]

    prompt = f"Question: {question}\n"
    letters = ["A", "B", "C", "D"]

    for i, choice in enumerate(choices):
        prompt += f"{letters[i]}. {choice}\n"

    prompt += "Answer:"
    return prompt

if __name__ == "__main__":
    print("Loading MMLU dataset...")

    if SUBJECT:
        dataset = load_dataset(DATASET_NAME, SUBJECT, split=SPLIT)
    else:
        dataset = load_dataset(DATASET_NAME, split=SPLIT)

    if MAX_SAMPLES:
        dataset = dataset.select(range(MAX_SAMPLES))

    print(f"Loaded {len(dataset)} samples")

    os.makedirs("data", exist_ok=True)
    os.makedirs("gold", exist_ok=True)

    print("Writing raw prompt file...")

    gold_answers = []

    with open(RAW_DATASET, "w") as f:
        for idx, sample in enumerate(dataset):
            prompt = format_prompt(sample)
            f.write(prompt.replace("\n", " ") + "\n") # one-line per sample

            gold_answers.append({
                "id": idx,
                "answer": sample["answer"] # integer index 0-3
            })

    print("Writing gold labels...")

    with open(GOLD_FILE, "w") as f:
        json.dump(gold_answers, f, indent=2)