import sys
import os
import json
import argparse
from tqdm import tqdm
import torch

# Add MMLongBench code to path
sys.path.append(os.path.abspath("code/MMLongBench"))

from data import load_icl_er
from vlm_model.openai_model import OpenAIModel
from src.recursive_model import RecursiveICLModel

class MockArgs:
    def __init__(self, test_file, image_root, max_samples=None):
        self.test_file_root = os.path.dirname(test_file)
        # The loader expects path relative to test_file_root, so we pass just the filename
        self.dataset_filename = os.path.basename(test_file) 
        self.image_file_root = image_root
        self.preprocessing_num_workers = 1
        self.seed = 42
        self.max_test_samples = max_samples

def run_eval(model, dataset, output_file):
    results = []
    correct = 0
    total = 0
    
    hf_data = dataset['data']
    print(f"Running evaluation on {len(hf_data)} samples...")
    
    for i in tqdm(range(len(hf_data))):
        # sample in dataset['data'] is a dict with context, question, image_list, answer
        raw_sample = hf_data[i]
        
        try:
            inputs = model.prepare_inputs(raw_sample, dataset)
            
            # Generate
            output = model.generate(inputs)
            prediction = output["output"]
            
            # Post-process (simple exact match check for now, or use the provided one)
            # The label is usually "label: X".
            # prediction might be "label: X" or just "X".
            
            gt = raw_sample["answer"]
            
            is_correct = str(gt).lower() in prediction.lower()
            if is_correct:
                correct += 1
            total += 1
            
            result_item = {
                "id": i,
                "prediction": prediction,
                "ground_truth": gt,
                "correct": is_correct,
                "input_len": output["input_len"],
                "output_len": output["output_len"]
            }
            results.append(result_item)
            
            # Intermediate save
            if i % 5 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="recursive", choices=["recursive", "baseline"])
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=5)
    args = parser.parse_args()

    # Configuration
    data_path = "datasets/MMLongBench/mmlb_data/ICL/inat2021_K16.json"
    image_root = "datasets/MMLongBench/mmlb_image"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    mock_args = MockArgs(data_path, image_root, max_samples=args.max_samples)
    
    # Load Dataset
    print("Loading dataset...")
    # load_icl_er takes (args, path) where path is relative to test_file_root
    dataset = load_icl_er(mock_args, mock_args.dataset_filename, max_test_samples=args.max_samples)
    
    # Initialize Model
    if args.model_type == "recursive":
        print("Initializing Recursive Model...")
        model = RecursiveICLModel("gpt-4o", chunk_size=args.chunk_size)
        output_file = os.path.join(output_dir, f"recursive_results_n{args.max_samples}.json")
    else:
        print("Initializing Baseline Model...")
        model = OpenAIModel("gpt-4o")
        output_file = os.path.join(output_dir, f"baseline_results_n{args.max_samples}.json")
        
    # Run
    acc = run_eval(model, dataset, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
