# Research Plan: Recursive Language Models for Cross-Modal Inference

## Research Question
Can a Recursive Language Model (RLM) framework, enhanced with Visual In-Context Learning (VICL) and recursive chunking, outperform standard Multimodal LLMs on long-context cross-modal reasoning tasks?

## Background and Motivation
Standard MLLMs (like GPT-4V, Gemini 1.5) handle long contexts well but often struggle with precise "needle" retrieval or complex reasoning scattered across many images or long documents. Recursive approaches (decomposing tasks into smaller steps or chunks) have shown promise in text. This research explores applying recursion to the visual domain, leveraging Visual In-Context Learning to handle visual analogies dynamically.

## Hypothesis Decomposition
1.  **H1 (Textual Recursion):** Recursive chunking and summarization improves performance on long-context text tasks compared to direct processing (validating the "Recursive" part).
2.  **H2 (Visual Recursion/ICL):** (If images available) A recursive approach that uses In-Context Learning for visual queries outperforms standard direct visual QA on long sequences.

## Proposed Methodology

### Approach: Recursive-VICL
We will implement a **Recursive Reasoning Agent** that:
1.  **Decomposes** a complex/long query into sub-queries.
2.  **Recursively processes** the input stream (text or image list) in chunks.
3.  **Accumulates** evidence using a "scratchpad" or state.
4.  **Synthesizes** the final answer.

For the **Visual** component (if data permits), we will use the `ICL` (In-Context Learning) subset of MMLongBench, where the model must learn from visual examples in the context.

### Experimental Steps
1.  **Environment Setup:** Create `uv` venv, install `transformers`, `torch`, `openai`/`anthropic`/`google` SDKs.
2.  **Data Prep:**
    *   Use `MMLongBench` text datasets (NIAH, DocQA) immediately.
    *   Attempt to download `3_icl_image.tar.gz` for the Visual ICL task.
3.  **Implementation:**
    *   **Baseline:** Direct API call with full context (limited by token window or relying on model's native long-context).
    *   **Recursive Method:** A function that splits context, processes chunks with a local or API model (summarizing/extracting), and aggregates.
4.  **Experiment:** Run both methods on a subset of the data (e.g., 20-50 samples) to measure Accuracy.

### Baselines
*   **Direct-Long-Context:** Feeding the entire long context to the LLM (e.g., GPT-4o or Gemini-1.5-Pro).
*   **Sliding Window (Implicit):** Standard RAG or simple chunking without recursion (if time permits).

### Evaluation Metrics
*   **Accuracy:** Exact Match or LLM-as-Judge scoring (as defined in MMLongBench).
*   **Cost/Latency:** Token usage comparison.

## Expected Outcomes
We expect the Recursive method to show higher accuracy on tasks requiring information distributed across the context (e.g., counting, reasoning), potentially at a higher token cost.

## Timeline
*   **Phase 1 (Planning):** 20 min (Done)
*   **Phase 2 (Setup & Data):** 20 min (Env setup, download check)
*   **Phase 3 (Implementation):** 60 min (Coding Baseline and Recursive Agent)
*   **Phase 4 (Experimentation):** 60 min (Running on NIAH/ICL)
*   **Phase 5 (Analysis):** 30 min (Results parsing, plotting)
*   **Phase 6 (Documentation):** 30 min (Report writing)

## Potential Challenges
*   **Data Availability:** Image download might be slow. Fallback: Focus on Textual Recursion on `NIAH-text`.
*   **API Limits:** Long contexts consume many tokens. Mitigation: Use small subsets (10-20 samples) and `gemini-1.5-flash` or `gpt-4o-mini` for development.
*   **Complexity:** Implementing full RVP is complex. Mitigation: Simplified "Recursive Summarization" instead of full program generation.

## Success Criteria
*   Successfully running the MMLongBench evaluation harness.
*   Obtaining comparative results between Baseline and Recursive method on at least one task (Text or Image).
