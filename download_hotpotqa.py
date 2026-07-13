from datasets import load_dataset

print("Starting download...")
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
print(dataset)

dataset.save_to_disk("./data/hotpotqa_distractor")
print("Saved to ./data/hotpotqa_distractor")