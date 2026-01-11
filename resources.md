# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Recursive Language Models for Cross-Modal Inference".

### Papers
Total papers: 5

| Title | Year | File |
|-------|------|------|
| Recursive Visual Attention | 2018 | `papers/1812.02664_RecursiveVisualAttention.pdf` |
| Recursive Visual Programming | 2023 | `papers/2312.02249_RecursiveVisualProgramming.pdf` |
| Visual In-Context Learning | 2024 | `papers/2402.11574_VisualInContext.pdf` |
| MMLongBench | 2025 | `papers/2505.10610_MMLongBench.pdf` |
| VReST | 2025 | `papers/2506.08691_VReST.pdf` |

### Datasets
Total datasets: 1 (Primary)

| Name | Source | Size | Location |
|------|--------|------|----------|
| MMLongBench | HuggingFace | ~10GB (Text+Meta) | `datasets/MMLongBench/` |

**Note**: Image data for MMLongBench is not fully downloaded to save space. Scripts to download it are available in `code/MMLongBench/scripts/`.

### Code Repositories
Total repositories: 3

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| MMLongBench | github.com/EdinburghNLP/MMLongBench | Benchmark & Data | `code/MMLongBench/` |
| RVP | github.com/para-lost/RVP | Recursive Reasoning | `code/RVP/` |
| DINOv | github.com/UX-Decoder/DINOv | Visual ICL | `code/DINOv/` |

### Resource Gathering Notes
- **Strategy**: Targeted search for the papers mentioned in the hypothesis and the specific benchmark `MMLongBench`.
- **Completion**: All key components (Methodology: RVP, ICL; Evaluation: MMLongBench) have been located and cloned/downloaded.
- **Next Steps**: The experiment runner should use `code/MMLongBench` to set up the evaluation environment and integrate `RVP` and `DINOv` logic.
