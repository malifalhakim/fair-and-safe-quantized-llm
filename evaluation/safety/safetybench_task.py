import numpy as np
from datasets import Dataset

def process_docs(dataset):
    """
    Processes the dataset for the Jigsaw task.
    """
    processed_docs_list = []
    for doc in dataset:
        question_prompt = f"Question: {doc['question'].strip()}\nOptions:\n"
        options = [opt.strip() for opt in doc["options"]]
        for i, option in enumerate(options):
            question_prompt += f"{chr(65 + i)}. {option}\n"
        question_prompt += "Answer:"

        options_characters = [chr(65 + i) for i in range(len(options))]


        processed_docs_list.append({
            "id": doc["id"],
            "question": question_prompt,
            "options": options_characters,
            "answer": options_characters[0] # Dummy answer since the metric will be evaluated later
        })
    return Dataset.from_list(processed_docs_list)