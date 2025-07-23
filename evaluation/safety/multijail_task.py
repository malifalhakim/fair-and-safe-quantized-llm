import json
import argparse

from datasets import load_dataset

from walledeval.data import HuggingFaceDataset
from walledeval.types import Prompt
from walledeval.judge import WalledGuardJudge
from walledeval.llm import HF_LLM

def prepare_dataset(lang_code: str = "en"):
    """
    Prepares the MultiJail dataset for evaluation.
    """
    dataset = load_dataset("DAMO-NLP-SG/MultiJail", split="train")
    raw_data = []
    for data in dataset:
        raw_data.append(data[lang_code])
        
    return HuggingFaceDataset[Prompt].from_list("mysafetydata", raw_data)

def prepare_models(model_name: str):
    """
    Prepares the models for evaluation.
    """
    generator = HF_LLM(model_name, device_map="auto")

    judge = WalledGuardJudge(
        device_map="auto"
    )

    return generator, judge

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
    dataset = prepare_dataset(args.lang_code)
    generator, judge = prepare_models(args.model_name)

    logs = []
    total_wrong = 0
    for data in dataset:
        response = generator(data.prompt)
        judge_output, score = judge(response)

        logs.append({
            "prompt": data.prompt,
            "response": response,
            "judge_output": judge_output,
            "is_safe": score
        })

        if not score:
            total_wrong += 1

    save_logs(logs, args.output_file)
    print(f"Total unsafe responses: {total_wrong}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MultiJail task")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--lang_code", type=str, default="en", help="Language code for the dataset")
    parser.add_argument("--output_file", type=str, default="multijail_results.jsonl", help="File to save evaluation logs")
    
    args = parser.parse_args()
    main(args)