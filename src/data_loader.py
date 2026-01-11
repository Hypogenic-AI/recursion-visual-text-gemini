import json
import os

def load_mmlongbench_data(file_path, limit=None):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line {i}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    
    return data

def get_dataset_files(base_dir="datasets/MMLongBench/mmlb_data"):
    # Helper to list available dataset files
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                files.append(os.path.join(root, filename))
    return files
