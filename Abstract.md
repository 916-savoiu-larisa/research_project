## Abstract — Why Read This Paper?

This chapter introduces an emotion-aware chatbot designed for real-time interaction under strict data and resource constraints. The system couples a fine-tuned BERT classifier (8 emotions: joy, sadness, anger, fear, surprise, disgust, love, neutral) with a keyword-first override and template-driven responses drawn from a curated CSV. The goal is to deliver stable, transparent behavior when only a handful of labeled examples (64 utterances) are available, while keeping latency low enough for interactive use on CPU.

### Problem and Motivation
Emotion recognition in text is a cornerstone for empathetic conversational agents, yet performance often collapses when labeled data are scarce. Industrial-scale solutions (e.g., large empathetic agents) achieve robustness through massive corpora and heavy compute, but they are difficult to reproduce and unsuitable for lightweight deployments. This work addresses the need for an accessible, modular pipeline that preserves responsiveness and interpretability in the small-data regime.

### Approach
The system implements a hybrid inference pipeline: (1) keyword override prioritizes high-precision lexical cues; (2) BERT-based classification handles open-text cases; (3) confidence-aware switching routes low-confidence predictions back to lexical rules; (4) negative/positive reconciliation reduces polarity flips. Responses are selected deterministically per emotion from the CSV, with light randomization within each class to avoid repetition. A Flask REST API exposes `/chat` and `/ping`, enabling straightforward integration.

### Original Contributions
1) **Hybrid stability**: Combining lexical overrides with neural inference to mitigate small-data brittleness.  
2) **Deterministic mapping**: Automatic label and response alignment from the curated CSV, ensuring auditable behavior.  
3) **Portable serving**: A minimal Flask API that runs on CPU without specialized hardware.  
4) **Empirical profile**: CPU latency ~39 ms/prediction (~25 pred/s), response selection ~0.20 ms, model size ~440 MB (BERT-base), demonstrating real-time feasibility.

### Findings and Implications
The measured latency confirms interactive performance on CPU, and the keyword fallback materially stabilizes outputs despite modest accuracy driven by limited data. The chapter argues that empathetic interaction is attainable with transparent, reproducible components, setting a baseline for future extensions such as larger datasets, distilled models, and multilingual coverage.

