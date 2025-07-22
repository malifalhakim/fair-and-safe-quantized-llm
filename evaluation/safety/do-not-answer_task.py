import json
import argparse

from datasets import load_dataset

from walledeval.data import HuggingFaceDataset
from walledeval.types import Prompt
from walledeval.judge import WalledGuardJudge
from walledeval.llm import HF_LLM
from walledeval.prompts import PromptTemplate

from transformers import pipeline



def prepare_dataset():
    """
    Prepares the XSTest dataset for evaluation.
    """
    dataset = load_dataset("LibrAI/do-not-answer", split="train")
    raw_data = []
    for data in dataset:
        raw_data.append(data['question'])
    
    return HuggingFaceDataset[Prompt].from_list("mysafetydata", raw_data)

def prepare_models(model_name: str):
    """
    Prepares the models for evaluation.
    """
    generator = HF_LLM.from_pretrained(model_name, device_map="auto")

    judge = pipeline("text-classification", model="LibrAI/longformer-harmful-ro")
    classifier = pipeline("text-classification", model="LibrAI/longformer-action-ro")

    return generator, judge, classifier

def save_logs(logs, output_file):
    """
    Saves the logs to jsonl format.
    """
    with open(output_file, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

def main(args):
    """
    Main function to run the evaluation.
    """
    dataset = prepare_dataset()
    generator, judge, classifier = prepare_models(args.model_name)

    logs = []
    total_wrong = 0
    for data in dataset:
        template = PromptTemplate()
        prompt = template.format(data.prompt)

        response = generator(prompt)
        is_harmful = judge(response)[0]['label']
        type_of_action = classifier(response)[0]['label']

        logs.append({
            "prompt": prompt,
            "response": response,
            "is_harmful": is_harmful,
            "type_of_action": type_of_action
        })

        if is_harmful:
            total_wrong += 1

    save_logs(logs, args.output_file)
    print(f"Total unsafe responses: {total_wrong}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Do Not Answer task")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--output_file", type=str, default="do_not_answer_results.jsonl", help="File to save evaluation logs")
    
    args = parser.parse_args()
    main(args)