# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size.

## Dataset 1: MMLongBench

### Overview
- **Source**: HuggingFace (ZhaoweiWang/MMLongBench)
- **Code Repository**: https://github.com/EdinburghNLP/MMLongBench
- **Task**: Multimodal Long-Context Understanding (NIAH, Retrieval, Reasoning, ICL, etc.)
- **Format**: JSONL files (text/metadata) + Image files (separate download)
- **Location**: `datasets/MMLongBench/`

### Download Instructions

**Text Data (Downloaded):**
```bash
wget https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/0_mmlb_data.tar.gz
tar -xzvf 0_mmlb_data.tar.gz
```

**Image Data (Not Downloaded - Large):**
To download the image data, use the script in `code/MMLongBench/scripts/download_image_data.sh` or run:
```bash
# This will download ~50GB+ of images
bash code/MMLongBench/scripts/download_image_data.sh
```

### Loading the Dataset

The text data is in `datasets/MMLongBench/mmlb_data/`. Each subdirectory corresponds to a task (e.g., `NIAH`, `ICL`, `documentQA`).

Example loading in Python:
```python
import json
import os

data_path = "datasets/MMLongBench/mmlb_data/NIAH/reasoning-image_test_K4_dep6.jsonl"
with open(data_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        print(item)
        break
```

### Notes
- Only text/metadata is currently present in `datasets/MMLongBench`.
- Image paths in JSONL files typically point to local paths that need to be populated by downloading the image data.
