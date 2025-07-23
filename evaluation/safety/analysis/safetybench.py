import json

def get_prediction_from_responses(resp: list) -> float:
    """
    Determines the prediction based on the highest probability response.
    """
    pred = -1
    max_log_likelihood = float('-inf')

    for i , resp in enumerate(resp):
        if resp and len(resp) > 0 and len(resp[0]) > 0:
            try:
                log_likelihood = float(resp[0][0])
                if log_likelihood > max_log_likelihood:
                    max_log_likelihood = log_likelihood
                    pred = i
            except (ValueError, IndexError):
                continue

    return pred

def convert_jsonl_to_json(input_file: str, output_file: str):
    """
    Converts a JSONL file to a JSON file.
    """
    data_submission = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            doc = data.get("doc", {})

            data_id = str(doc.get('id', ''))
            pred = int(get_prediction_from_responses(data.get('resps', [])))

            data_submission[data_id] = pred
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_submission, f, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSONL to JSON")
    parser.add_argument("input_file", type=str, help="Input JSONL file")
    parser.add_argument("output_file", type=str, help="Output JSON file")

    args = parser.parse_args()

    convert_jsonl_to_json(args.input_file, args.output_file)
            