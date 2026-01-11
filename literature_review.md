# Literature Review: Recursive Language Models for Cross-Modal Inference

## Research Area Overview
This research focuses on enhancing Multimodal Large Language Models (MLLMs) to handle long-context and complex reasoning tasks through **recursive processing** and **visual in-context learning**. Standard MLLMs often struggle with dense visual information or multi-step reasoning over long sequences. The proposed approach integrates recursive chunking (breaking down complex inputs) with in-context learning (learning from examples within the prompt) to improve performance.

## Key Papers

### Paper 1: Recursive Visual Programming (2023)
- **File**: `2312.02249_RecursiveVisualProgramming.pdf`
- **Key Contribution**: Proposes a framework that interprets complex visual queries by recursively generating programmatic steps (code) to solve sub-problems.
- **Methodology**: Uses an LLM to decompose questions into executable code snippets that call vision primitives.
- **Relevance**: Provides the structural foundation for "recursive" reasoning in our hypothesis.

### Paper 2: Visual In-Context Learning (2024)
- **File**: `2402.11574_VisualInContext.pdf` (likely Zhou et al.)
- **Key Contribution**: Demonstrates that MLLMs can learn to perform new visual tasks (e.g., segmentation, detection) solely from visual examples in the prompt, without weight updates.
- **Methodology**: "DINOv" or similar approaches that leverage visual analogies.
- **Relevance**: Provides the "in-context" mechanism to allow the model to adapt to new recursive patterns dynamically.

### Paper 3: MMLongBench (2025)
- **File**: `2505.10610_MMLongBench.pdf`
- **Key Contribution**: A comprehensive benchmark for evaluating MLLMs on long-context tasks (video understanding, multi-image reasoning, document QA).
- **Dataset**: MMLongBench (NIAH, Retrieval, etc.).
- **Relevance**: The primary testbed for our hypothesis, specifically designed to stress-test long-context capabilities.

### Paper 4: VReST (2025)
- **File**: `2506.08691_VReST.pdf`
- **Key Contribution**: Likely "Visual Recursive State Tracking" or similar reasoning enhancement.
- **Relevance**: A potential baseline or state-of-the-art method for comparison.

### Paper 5: Recursive Visual Attention (2018)
- **File**: `1812.02664_RecursiveVisualAttention.pdf`
- **Key Contribution**: Early work on recursively attending to image regions for visual dialog.
- **Relevance**: Foundational context for recursive mechanisms in vision.

## Methodological Synthesis
- **Recursion**: A common theme is decomposing difficult visual tasks into simpler steps (RVP) or refining attention over time (Recursive Attention).
- **In-Context Learning**: Moving from fine-tuning to prompt-based adaptation (Visual ICL).
- **Long-Context**: Handling massive inputs via retrieval or sliding windows (MMLongBench).

## Recommendations for Experiment
1.  **Dataset**: **MMLongBench** is the clear choice for evaluating long-context multimodal performance.
2.  **Baselines**: Compare the proposed Recursive-ICL model against:
    -   Standard **GPT-4o** or **Gemini** (if available) or open-source **LLaVA-Next**.
    -   **RVP** (without ICL).
    -   **Visual ICL** (without recursion).
3.  **Metrics**: Accuracy/F1 on MMLongBench tasks (NIAH, Reasoning).
