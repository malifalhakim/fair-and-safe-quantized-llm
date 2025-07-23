import json
import argparse

from datasets import load_dataset

from walledeval.data import HuggingFaceDataset
from walledeval.types import SafetyPrompt
from walledeval.judge import WalledGuardJudge
from walledeval.llm import HF_LLM


def prepare_dataset():
    """
    Prepares the XSTest dataset for evaluation.
    """
    dataset = load_dataset("walledai/XSTest", split="test")
    raw_data = []
    for data in dataset:
        raw_data.append({
            "prompt": data["prompt"],
            "label": data["label"],
        })
    
    return HuggingFaceDataset[SafetyPrompt].from_list("mysafetydata", raw_data)

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
    dataset = prepare_dataset()
    generator, judge = prepare_models(args.model_name)

    logs = []
    total_correct = 0
    for data in dataset:
        response = generator(data.prompt)
        judge_output, score = judge(response)
        
        logs.append({
            "prompt": data.prompt,
            "label": data.label,
            "response": response,
            "judge_output": judge_output,
            "score": score,
            "is_correct": judge_output == data.label
        })

        if judge_output == data.label:
            total_correct += 1
    
    accuracy = total_correct / len(dataset)
    print(f"Accuracy: {accuracy:.2f}")

    save_logs(logs, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XSTest evaluation.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--output_file", type=str, default="xstest_results.jsonl", help="File to save the results.")

    args = parser.parse_args()
    main(args)

