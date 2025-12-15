## Description of the Original Approach — How Was the Problem Solved?

This chapter presents the proposed solution to the emotion-aware chatbot problem, detailing the hybrid inference architecture, mathematical formalization, algorithms, and the original aspects that distinguish this approach from existing literature.

### System Architecture Overview

The proposed system implements a modular pipeline consisting of four primary components:

1. **Input Preprocessing**: Text normalization and tokenization
2. **Hybrid Emotion Classifier**: Keyword-first override with BERT-based neural classification
3. **Confidence-Aware Decision Logic**: Routes low-confidence predictions to lexical fallback
4. **Template-Based Response Selection**: Deterministic emotion-to-response mapping from curated CSV

### Mathematical/Formal Model

#### Problem Formulation

Let \( \mathcal{T} \) be the set of all possible text inputs, and \( \mathcal{E} = \{e_1, e_2, \ldots, e_k\} \) be the set of \( k \) discrete emotion classes (in our case, \( k = 8 \): joy, sadness, anger, fear, surprise, disgust, love, neutral).

The emotion classification task is defined as a function:
\[
f: \mathcal{T} \rightarrow \mathcal{E}
\]

Given a text input \( t \in \mathcal{T} \), the system must predict the most likely emotion \( e \in \mathcal{E} \).

#### BERT-Based Classification Model

The neural component uses a fine-tuned BERT encoder. Formally, for input text \( t \), the model:

1. **Tokenization**: Maps \( t \) to a sequence of subword tokens:
   \[
   \mathbf{t} = [t_1, t_2, \ldots, t_n] = \text{Tokenizer}(t)
   \]
   where \( n \leq 128 \) (max sequence length).

2. **Embedding**: BERT produces contextualized embeddings:
   \[
   \mathbf{H} = \text{BERT}(\mathbf{t}) \in \mathbb{R}^{n \times d}
   \]
   where \( d = 768 \) for BERT-base.

3. **Classification Head**: The [CLS] token embedding \( \mathbf{h}_{[CLS]} \in \mathbb{R}^d \) is passed through a linear layer:
   \[
   \mathbf{z} = \mathbf{W} \mathbf{h}_{[CLS]} + \mathbf{b} \in \mathbb{R}^k
   \]
   where \( \mathbf{W} \in \mathbb{R}^{k \times d} \) and \( \mathbf{b} \in \mathbb{R}^k \) are learned parameters.

4. **Probability Distribution**: Softmax normalization yields class probabilities:
   \[
   P(e_i | t) = \frac{\exp(z_i)}{\sum_{j=1}^{k} \exp(z_j)}
   \]

5. **Prediction**: The predicted emotion is:
   \[
   e_{\text{model}} = \arg\max_{e_i \in \mathcal{E}} P(e_i | t)
   \]

#### Hybrid Classification Function

The hybrid system combines keyword matching with neural inference:

\[
f_{\text{hybrid}}(t) = \begin{cases}
f_{\text{keyword}}(t) & \text{if } f_{\text{keyword}}(t) \neq \emptyset \\
e_{\text{model}} & \text{if } \max P(e_i | t) \geq \theta \text{ and } f_{\text{keyword}}(t) = \emptyset \\
f_{\text{keyword}}(t) & \text{if } \max P(e_i | t) < \theta
\end{cases}
\]

where:
- \( f_{\text{keyword}}: \mathcal{T} \rightarrow \mathcal{E} \cup \{\emptyset\} \) is the keyword matching function
- \( \theta = 0.3 \) is the confidence threshold
- Priority is given to positive emotions (joy, love) to prevent false negatives

#### Keyword Matching Function

The keyword override function is defined as:

\[
f_{\text{keyword}}(t) = \begin{cases}
e_i & \text{if } \exists k \in K_{e_i} \text{ such that } k \in \text{lower}(t) \\
\emptyset & \text{otherwise}
\end{cases}
\]

where \( K_{e_i} \) is the set of keywords/phrases associated with emotion \( e_i \), and the matching is performed in priority order (positive emotions first).

#### Response Selection Function

Given detected emotion \( e \), the response selection function maps to a response template:

\[
r = \text{SelectResponse}(e) = \begin{cases}
\text{RandomChoice}(\mathcal{R}_e) & \text{if } \mathcal{R}_e \neq \emptyset \\
\text{RandomChoice}(\mathcal{T}_e) & \text{otherwise}
\end{cases}
\]

where:
- \( \mathcal{R}_e \) is the set of responses from CSV for emotion \( e \)
- \( \mathcal{T}_e \) is the set of fallback templates for emotion \( e \)

### Proposed Algorithms

#### Algorithm 1: Hybrid Emotion Classification

```
Input: text t, model M, tokenizer T, keyword sets K, confidence threshold θ
Output: predicted emotion e

1. // Priority: check positive emotions first
2. for emotion e in [joy, love]:
3.     for keyword k in K[e]:
4.         if k in lower(t):
5.             return e
6. 
7. // Check other emotions
8. for emotion e in [sadness, anger, fear, surprise, disgust]:
9.     for keyword k in K[e]:
10.         if k in lower(t):
11.             return e
12. 
13. // Neural model prediction
14. tokens = T(t, max_length=128, truncation=True, padding=True)
15. logits = M(tokens)
16. probs = softmax(logits)
17. confidence = max(probs)
18. e_model = argmax(probs)
19. 
20. // Confidence check
21. if confidence < θ:
22.     return keyword_override(t, K)  // fallback
23. 
24. // Validate negative predictions
25. if e_model in [sadness, anger, fear, disgust]:
26.     e_keyword = keyword_override(t, K)
27.     if e_keyword in [joy, love]:
28.         return e_keyword  // prefer positive
29. 
30. return e_model
```

#### Algorithm 2: Response Selection

```
Input: emotion e, CSV responses R, fallback templates T
Output: response string r

1. if R[e] exists and |R[e]| > 0:
2.     return RandomChoice(R[e])
3. else:
4.     return RandomChoice(T[e])
```

#### Algorithm 3: Training Procedure

```
Input: dataset D, base model M_base, tokenizer T, epochs E, batch size B
Output: fine-tuned model M

1. // Data preparation
2. (D_train, D_val, D_test) = Split(D, ratios=[0.7, 0.15, 0.15], seed=42)
3. label_map = GenerateLabelMap(D)
4. 
5. // Tokenization
6. for split in [train, val, test]:
7.     D_split = Tokenize(D_split, T, max_length=128)
8. 
9. // Model initialization
10. M = AutoModelForSequenceClassification.from_pretrained(
11.     M_base, num_labels=|label_map|, id2label=label_map
12. )
13. 
14. // Training
15. for epoch in 1..E:
16.     for batch in Batches(D_train, size=B):
17.         loss = CrossEntropy(M(batch.inputs), batch.labels)
18.         loss.backward()
19.         optimizer.step()
20.     
21.     // Evaluation
22.     metrics = Evaluate(M, D_val)
23.     if metrics.f1 > best_f1:
24.         SaveModel(M)
25. 
26. return M
```

### Original Aspects Different from Literature

The proposed approach introduces several original contributions that distinguish it from existing emotion-aware chatbot systems:

#### 1. **Hybrid Inference with Priority-Based Keyword Override**

Unlike systems that rely solely on neural models or rule-based approaches, this work combines both with a priority mechanism that checks positive emotions first. This prevents false negatives where positive expressions (e.g., "I'm so happy!") are misclassified as negative emotions due to limited training data. The keyword override acts as a high-precision filter before neural inference, reducing computational cost and improving reliability.

**Difference from literature**: Most hybrid systems use neural models as primary and keywords as fallback. This work reverses the priority, using keywords as the first line of defense, which is particularly effective in small-data regimes.

#### 2. **Confidence-Aware Polarity Reconciliation**

The system implements a double-check mechanism: if the neural model predicts a negative emotion but keyword matching suggests a positive emotion, the system overrides the prediction. This addresses a common failure mode in small-data training where positive expressions are underrepresented.

**Difference from literature**: Existing confidence-based systems typically only use threshold-based fallback. This work adds polarity validation, explicitly checking for emotion polarity mismatches.

#### 3. **Deterministic Response Mapping from Structured Data**

Responses are loaded directly from the training CSV, ensuring that the emotion-to-response mapping is transparent, auditable, and automatically aligned with the label space. The system extracts responses from the same dataset used for training, maintaining consistency.

**Difference from literature**: Many systems use separate response generation modules (e.g., GPT-based generators) that are opaque. This work uses deterministic template selection from a curated, documented source.

#### 4. **Reproducible Pipeline with Fixed Splits and Seeds**

All randomness is controlled via fixed seeds (seed=42), and data splits are deterministic. This ensures that training, evaluation, and inference use the same preprocessing pipeline and label mappings, enabling exact reproducibility.

**Difference from literature**: While reproducibility is a best practice, this work explicitly documents and enforces it at every stage (tokenization, splitting, training, inference).

#### 5. **CPU-Optimized Real-Time Inference**

The system is designed and validated for CPU-only inference, achieving ~39 ms latency without GPU acceleration. This makes the system accessible for deployment in resource-constrained environments.

**Difference from literature**: Most transformer-based emotion systems assume GPU availability. This work explicitly targets CPU deployment and validates performance accordingly.

### Implementation Details

#### Training Configuration

- **Base Model**: `bert-base-uncased` (110M parameters)
- **Tokenizer**: BERT WordPiece tokenizer (vocab size: 30,522)
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 8 (training), 16 (evaluation)
- **Learning Rate**: Default AdamW optimizer (2e-5)
- **Epochs**: 3 (default)
- **Evaluation Metric**: Macro-averaged F1 score
- **Data Split**: 70% train, 15% validation, 15% test (deterministic, seed=42)

#### Inference Configuration

- **Device**: CPU (PyTorch CPU mode)
- **Confidence Threshold**: 0.3
- **Keyword Matching**: Case-insensitive substring matching
- **Response Selection**: Random choice within emotion class

#### System Constraints

- **Sequence Length**: Maximum 128 tokens (truncation for longer inputs)
- **Memory**: ~440 MB for model weights
- **Latency Target**: < 50 ms per prediction (achieved: ~39 ms)
- **Throughput**: > 20 predictions/second (achieved: ~25 pred/s)

### Summary

The proposed approach addresses the small-data, low-resource emotion classification problem through a hybrid architecture that prioritizes interpretable keyword matching while leveraging BERT for open-text cases. The system's original contributions lie in its priority-based override mechanism, confidence-aware polarity validation, deterministic response mapping, and CPU-optimized design. These features make the system transparent, reproducible, and accessible for research and lightweight deployment scenarios.

