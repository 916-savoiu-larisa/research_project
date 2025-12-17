# Emotion Chatbot System - Code Documentation and Performance Analysis

## Table of Contents
1. [System Overview](#system-overview)
2. [Code Architecture](#code-architecture)
3. [Component Details](#component-details)
4. [Performance Testing](#performance-testing)
5. [Chapter Summaries](#chapter-summaries)
   - [Chapter 5 Summary](#chapter-5-summary)
   - [Chapter 6 Summary](#chapter-6-summary)
   - [Chapter 7 Summary](#chapter-7-summary)

---

## System Overview

This project implements an **emotion-aware chatbot system** that detects emotions in user text and generates appropriate empathic responses. The system uses a fine-tuned BERT-based model (`bert-base-uncased`) for emotion classification across 8 emotion categories: `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `love`, and `neutral`.

### Key Features
- **Emotion Classification**: Fine-tuned BERT model for multi-class emotion detection
- **Hybrid Prediction System**: Combines neural model predictions with keyword-based fallback for improved accuracy
- **Context-Aware Responses**: Selects empathic responses from a curated dataset based on detected emotions
- **RESTful API**: Flask-based web service for real-time emotion detection and response generation
- **Performance Optimized**: Fast inference with keyword matching for common cases

---

## Code Architecture

The codebase is organized into modular components:

```
research_project/
├── src/
│   ├── models/          # Model training and evaluation
│   │   ├── train_classifier.py    # Fine-tune BERT for emotion classification
│   │   └── evaluate.py            # Evaluate model performance
│   ├── chatbot/         # Chatbot application
│   │   ├── app.py                 # Flask REST API server
│   │   └── response_selector.py   # Emotion detection and response selection
│   └── utils/           # Utility functions
│       ├── dataset.py             # Data loading and preprocessing
│       └── metrics.py             # Evaluation metrics
├── data/                # Dataset files
│   └── emotion_chatbot_samples.csv
├── models/              # Trained model checkpoints
└── tests/               # Unit tests
```

---

## Component Details

### 1. Model Training (`src/models/train_classifier.py`)

**Purpose**: Fine-tunes a pre-trained BERT model for emotion classification.

**Key Functionality**:
- Loads emotion-labeled text data from CSV
- Tokenizes text using BERT tokenizer (max_length=128)
- Splits data into train/validation/test sets (70%/15%/15%)
- Fine-tunes `bert-base-uncased` using Hugging Face Transformers
- Saves model and tokenizer to specified output directory

**Training Configuration**:
- Base model: `bert-base-uncased`
- Batch size: 8 (train), 16 (eval)
- Epochs: 3 (default)
- Evaluation strategy: After each epoch
- Best model selection: Based on F1 score

**Usage**:
```bash
python src/models/train_classifier.py --data data/emotion_chatbot_samples.csv --output models/emotion_bert_small --epochs 3
```

### 2. Model Evaluation (`src/models/evaluate.py`)

**Purpose**: Evaluates trained model performance on test set.

**Key Functionality**:
- Loads trained model and tokenizer
- Evaluates on test split using same preprocessing pipeline
- Computes macro-averaged precision, recall, F1, and accuracy

**Metrics Computed**:
- Accuracy: Overall classification accuracy
- Precision (macro): Average precision across all emotion classes
- Recall (macro): Average recall across all emotion classes
- F1 Score (macro): Harmonic mean of precision and recall

**Usage**:
```bash
python src/models/evaluate.py --model models/emotion_bert_small --data data/emotion_chatbot_samples.csv
```

### 3. Response Selector (`src/chatbot/response_selector.py`)

**Purpose**: Core component that detects emotions and selects appropriate responses.

**Key Features**:

#### Emotion Prediction Pipeline:
1. **Keyword Override (Priority)**: Checks for emotion-indicating keywords first, prioritizing positive emotions to avoid false negatives
2. **Model Prediction**: Uses fine-tuned BERT model if no keyword match
3. **Confidence Check**: If model confidence < 0.3, falls back to keyword matching
4. **Negative Emotion Validation**: Double-checks if model predicts negative emotion but keywords suggest positive

#### Response Selection:
- **Primary**: Loads responses from CSV dataset grouped by emotion
- **Fallback**: Uses hardcoded templates if CSV responses unavailable
- **Randomization**: Selects random response from available options for variety

**Keyword Hints System**:
- Comprehensive keyword lists for each emotion
- Positive emotions (joy, love) checked first to prevent misclassification
- Negative emotions (sadness, anger, fear, disgust) checked second

**Usage**:
```python
selector = ResponseSelector(model_path='models/emotion_bert_small', csv_path='data/emotion_chatbot_samples.csv')
response = selector.select_response("I'm feeling great today!")
```

### 4. Flask API (`src/chatbot/app.py`)

**Purpose**: RESTful web service for emotion detection and response generation.

**Endpoints**:
- `GET /ping`: Health check endpoint (returns 'pong')
- `POST /chat`: Main endpoint for emotion detection and response
  - Request: `{"text": "user message"}`
  - Response: `{"response": "empathic response"}`

**Configuration**:
- Default host: `0.0.0.0` (all interfaces)
- Default port: `5000` (configurable via `PORT` environment variable)
- Model path: Configurable via `--model` argument

**Usage**:
```bash
python src/chatbot/app.py --model models/emotion_bert_small --csv data/emotion_chatbot_samples.csv
```

**Example API Call**:
```bash
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"text": "I just got promoted!"}'
```

### 5. Dataset Utilities (`src/utils/dataset.py`)

**Purpose**: Data loading and preprocessing for training.

**Key Functionality**:
- Loads CSV with columns: `text`, `label`
- Generates label mapping automatically from unique labels
- Splits data deterministically (70% train, 15% validation, 15% test)
- Tokenizes text using provided tokenizer
- Returns Hugging Face DatasetDict with train/validation/test splits

**Data Format**:
- CSV must contain `text` and `label` columns
- Labels are automatically mapped to integer indices
- Text is tokenized with truncation and padding to max_length

### 6. Metrics (`src/utils/metrics.py`)

**Purpose**: Computes evaluation metrics for model performance.

**Metrics**:
- **Accuracy**: Overall correct predictions
- **Precision (macro)**: Average precision across all classes
- **Recall (macro)**: Average recall across all classes
- **F1 Score (macro)**: Macro-averaged F1 score

Uses scikit-learn's `precision_recall_fscore_support` and `accuracy_score`.

---

## Performance Testing

Performance tests were conducted using `test_performance.py` script. Results below:

### Test Environment
- Model: `emotion_bert_small` (fine-tuned BERT-base-uncased)
- Dataset: 64 samples (56 training + 8 test)
- Test runs: 50-100 iterations per test
- Hardware: CPU-based inference

### 1. Inference Speed Test

**Results**:
- **Average inference time**: 39.31 ms per prediction
- **Standard deviation**: 2.27 ms
- **Min/Max time**: 36.00 ms / 46.01 ms
- **Throughput**: **25.44 predictions/second**

**Analysis**:
- Inference is fast enough for real-time chatbot applications
- Low variance indicates consistent performance
- Suitable for interactive use cases (< 50ms latency)

### 2. Response Selector Performance

**Results**:
- **Average response time**: 0.20 ms per response
- **Standard deviation**: 0.40 ms
- **Throughput**: **5,002 responses/second**

**Analysis**:
- Extremely fast response selection (keyword matching is instant)
- Most responses use keyword override, bypassing model inference
- System can handle high request volumes

### 3. Model Accuracy Test

**Results** (on test set):
- **Accuracy**: 33.33%
- **Precision (macro)**: 16.67%
- **Recall (macro)**: 19.44%
- **F1 Score (macro)**: 17.86%

**Analysis**:
- Lower accuracy expected due to small test set (9 samples)
- With only 64 total samples, model has limited training data
- Performance would improve significantly with larger dataset
- Keyword fallback system compensates for model limitations

### 4. Memory Usage

**Results**:
- **Model parameters**: 109,488,392 total parameters
- **Trainable parameters**: 109,488,392 (all parameters)
- **Model size**: ~440 MB (BERT-base-uncased standard size)

**Analysis**:
- Standard BERT-base model size
- Suitable for deployment on modern servers
- Can be optimized with model quantization if needed

### Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Inference Speed | 39.31 ms | Fast enough for real-time use |
| Response Selection | 0.20 ms | Extremely fast (keyword-based) |
| Throughput | 25.44 pred/s | Good for interactive applications |
| Model Accuracy | 33.33% | Limited by small dataset |
| Model Size | 109M params | Standard BERT-base size |

### Recommendations

1. **Dataset Expansion**: Increase training data to improve model accuracy
2. **Model Optimization**: Consider using smaller models (e.g., DistilBERT) for faster inference
3. **Caching**: Cache model predictions for repeated queries
4. **GPU Acceleration**: Use GPU for faster inference in production
5. **Hybrid Approach**: Current keyword + model approach is effective for small datasets

---

## Chapter Summaries

### Case Studies

Case Studies chapter provides a comprehensive literature review of emotion-aware conversational systems, contextualizing the proposed chatbot within the broader field of affective computing, natural language processing, and empathetic dialogue systems.

The chapter begins with the historical foundations of affective computing (Picard, Ekman, Cambria), establishing the psychological and computational principles that underpin emotion recognition in text. It reviews major emotion theories (basic emotions, dimensional models) and connects them to modern NLP-based classification approaches.

The chapter then surveys traditional approaches to emotion detection, such as keyword-based, lexicon-based, and machine-learning models, highlighting their limitations in handling linguistic ambiguity, contextual expressions, and long-term dependencies. This is followed by an extensive review of deep learning and transformer-based methods, with a focus on BERT, RoBERTa, GPT, and knowledge-enriched transformers. The literature consistently shows that transformer-based architectures yield state-of-the-art accuracy and robustness in emotion recognition tasks—an insight that motivates the choice of BERT in the current research.

Subsequently, the chapter analyzes emotion-aware dialogue systems, including AffectBot, XiaoIce, empathetic open-domain models (Rashkin et al.), and knowledge-enhanced conversational agents. Each system is compared to the current project along dimensions such as modality, emotional memory, context awareness, interpretability, and computational complexity. While XiaoIce and AffectBot represent large-scale, resource-intensive systems, the proposed chatbot prioritizes modularity, reproducibility, transparency, and deployment efficiency, making it more suitable for research environments and lightweight applications.

The chapter also compares traditional rule-based chatbots to modern emotion-aware systems, showing improvements in user satisfaction, engagement, and conversation length. These findings reinforce that emotional intelligence is a key factor in building meaningful, long-term human–AI interactions.

Finally, the chapter identifies research gaps: the need for transparent architectures, accessible implementations, lightweight deployment, and interpretable emotion-to-response mappings. The proposed system directly addresses these gaps with a modular BERT-based classifier, hybrid keyword fallback, and template-driven empathetic responses.

---

### Related work

Related work chapter develops a rigorous mathematical, architectural, and algorithmic model of the proposed emotion-aware chatbot. It formally defines all system components, data flows, and interactions, providing a reproducible and analytically grounded foundation for implementation.

The chapter begins by introducing the system architecture, which is modeled as a modular pipeline consisting of:

Input processing

Emotion recognition (BERT + keyword override)

Memory/context module

Response generation (template-based, emotion-aware)

Output formatting

Each module is defined using formal notation, including sets, functions, and mappings. The BERT classifier is modeled mathematically using embeddings, attention layers, feed-forward networks, and the cross-entropy optimization objective. The chapter also formalizes the hybrid prediction strategy, where rule-based keyword detection takes precedence in low-confidence scenarios, ensuring robustness despite the small dataset.

The data model and flow are described from dataset loading and tokenization to batch construction, training, inference, and multi-turn conversation management. The memory module is modeled as a state machine tracking conversation history and emotion transitions, enabling context-aware responses.

The chapter includes detailed algorithm definitions for:

Emotion classification

Keyword override logic

Context update

Template selection

Semantic similarity ranking (optional extension)

A complete performance model is then provided, calculating training time, inference latency, throughput, memory consumption, and computational complexity. This includes real-world performance results from your system:

~39ms average inference latency for BERT

0.20ms average response selection latency

Throughput of ~25 predictions/second

Model size ~440 MB
These findings confirm real-time feasibility even on CPU.

The chapter concludes with reproducibility mechanisms (fixed seeds, deterministic splits), system constraints (sequence length, context windows, memory limits), and assumptions (English-only, discrete emotion taxonomy, single-user sessions). Altogether, this chapter delivers a mathematically rigorous, fully specified model of the chatbot system.
---

### Modeling of Experimental System

Modeling of Experimental System chapter presents the full implementation, integration, and evaluation of the emotion-aware chatbot system, using the models described earlier. It translates the theoretical formalization into a functional software prototype built with Python, PyTorch, Hugging Face Transformers, Flask, and supporting utility modules.

The chapter begins with the software architecture, mirroring the chapter 6 model:

train_classifier.py for fine-tuning BERT

evaluate.py for performance assessment

response_selector.py implementing the hybrid classifier

app.py providing a REST API

utility modules for datasets and metrics

This modular architecture ensures clarity, maintainability, and extensibility.

Next, the chapter describes the training process, including tokenizer configuration, dataset splitting, hyperparameters (batch size, epochs, optimizer, learning rate), and the evaluation pipeline. The model is fine-tuned on your custom dataset of ~64 samples, with macro-averaged metrics showing modest accuracy due to the dataset size, but stable inference performance. The chapter highlights that the keyword fallback system substantially improves reliability beyond what raw accuracy metrics reflect.

The response selection system is presented as a lightweight yet effective mechanism for empathetic conversation. Emotion-specific response templates are loaded from a CSV file, enabling scalable and interpretable response strategies. The system is capable of real-time response generation with extremely low latency.

A performance evaluation section provides empirical measurements:

39.31 ms average model inference time

0.20 ms average response selector time

25.44 inferences / second throughput

109M parameters (standard BERT-base)

The Flask API is described along with usage examples, JSON schemas, error handling, and deployment notes.

The chapter ends with a discussion of limitations and future work, such as dataset enlargement, adopting smaller transformer models (DistilBERT) for optimized performance, enabling context-aware multi-turn emotion modeling, LLM-based response generation, and multi-language support.

Overall, Chapter 7 demonstrates that the implemented system is functional, efficient, modular, transparent, and well-aligned with the research objectives.
---

## Usage Examples

### Training a New Model
```bash
cd research_project
$env:PYTHONPATH="D:\Desktop\facultate\anul 3\Research project\research_project"
python src/models/train_classifier.py --data data/emotion_chatbot_samples.csv --output models/my_model --epochs 5
```

### Evaluating a Model
```bash
cd research_project
$env:PYTHONPATH="D:\Desktop\facultate\anul 3\Research project\research_project"
python src/models/evaluate.py --model models/emotion_bert_small --data data/emotion_chatbot_samples.csv
```

### Running the Chatbot API
```bash
cd research_project
$env:PYTHONPATH="D:\Desktop\facultate\anul 3\Research project\research_project"
python src/chatbot/app.py --model models/emotion_bert_small --csv data/emotion_chatbot_samples.csv
```

**Note**: Make sure you're in the `research_project` directory and set the PYTHONPATH so Python can find the `src` module. Alternatively, you can use:
```bash
cd research_project
python -m src.chatbot.app --model models/emotion_bert_small --csv data/emotion_chatbot_samples.csv
```

### Testing Performance
```bash
cd research_project
$env:PYTHONPATH="D:\Desktop\facultate\anul 3\Research project\research_project"
python test_performance.py --model models/emotion_bert_small --data data/emotion_chatbot_samples.csv --runs 100
```

### Testing via API
```bash
# Start the server
python src/chatbot/app.py

# In another terminal, test the API
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"text": "I feel sad today"}'
```

---

## Dependencies

Key dependencies (see `requirements.txt`):
- `transformers==4.57.1`: Hugging Face Transformers for BERT models
- `torch==2.8.0`: PyTorch for deep learning
- `flask==3.0.3`: Web framework for API
- `datasets==4.4.1`: Dataset handling
- `pandas==2.3.3`: Data manipulation
- `scikit-learn==1.6.1`: Evaluation metrics
- `sentence-transformers==5.1.2`: Semantic similarity (optional)

---

## Future Improvements

1. **Larger Dataset**: Collect more training samples for better model accuracy
2. **Context Awareness**: Track conversation history for multi-turn emotion detection
3. **Response Generation**: Use LLMs to generate dynamic responses instead of templates
4. **Multi-language Support**: Extend to multiple languages
5. **Emotion Intensity**: Predict emotion intensity levels (currently available in dataset)
6. **User Personalization**: Adapt responses based on user preferences
7. **Real-time Learning**: Fine-tune model based on user feedback

---

## Conclusion

This emotion chatbot system demonstrates a practical approach to emotion-aware conversational AI using fine-tuned BERT models. The hybrid keyword + neural model approach provides robust emotion detection even with limited training data. The system achieves good inference speed suitable for real-time applications, though model accuracy could be improved with a larger dataset.

The modular architecture allows for easy extension and improvement, making it a solid foundation for more advanced emotion-aware chatbot systems.

---