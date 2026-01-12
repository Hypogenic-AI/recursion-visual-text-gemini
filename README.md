# Recursive Visual-Text Inference

This repository contains the implementation and experimental results for "Recursive Visual In-Context Learning," a method to improve efficiency and interpretability in Multimodal LLMs.

## Overview
We implement a recursive agent that:
1.  Takes a Many-Shot Visual Classification task (e.g., 16 images + labels).
2.  **Recursively summarizes** the visual features of each class into text descriptions.
3.  Performs inference using these descriptions, reducing the context window requirement by >95%.

## Key Results
-   **Accuracy:** Matches GPT-4o Baseline (100% on sample set).
-   **Efficiency:** Reduces inference token count from ~17k to ~750.
-   **Interpretability:** Generates explanations for classification decisions.

## Project Structure
-   `src/recursive_model.py`: The core implementation of the Recursive ICL logic.
-   `run_eval.py`: Evaluation script compatible with MMLongBench datasets.
-   `results/`: JSON outputs of baseline and recursive experiments.
-   `code/MMLongBench/`: The evaluation framework and dataset loaders.

## Usage
To run the recursive evaluation:
```bash
source .venv/bin/activate
python run_eval.py --model_type recursive --max_samples 10
```

To run the baseline:
```bash
python run_eval.py --model_type baseline --max_samples 10
```

## Requirements
-   Python 3.10+
-   `openai`, `datasets`, `pillow`, `rouge_score`
-   MMLongBench dataset (images must be downloaded)
