from datasets import load_dataset

dataset = load_dataset("json", data_files="../data/wiki_categories.json", split="train")
dataset.push_to_hub("seven-tychi/wikipedia-categories")