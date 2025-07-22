import numpy as np
from datasets import Dataset

def process_docs(dataset):
    """
    Processes the dataset for the Jigsaw task.
    """
    processed_docs_list = []
    for doc in dataset:
        processed_docs_list.append({
            "id": doc["id"],
            "comment_text": doc["comment_text"],
            "toxic": 1 # Dummy label since the metric will be evaluated later 
        })
    return Dataset.from_list(processed_docs_list)


