# Emotion Chatbot Project

## Structure
- 'data/' - dataset CSVs
- 'src/' - source code
  - 'models/' - training and evaluation scripts
  - 'chatbot/' - flask prototype and response selector
  - 'utils/' - dataset helpers and metrics
- 'tests/' - unit/integration tests
- 'Dockerfile' - development container
- 'report/' - written chapters

## Quickstart
1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt  # or install transformers, torch, datasets, etc.
   ```
2. **Train / fine-tune (fast mode for smoke tests)**
   ```powershell
   $env:PYTHONPATH="D:\path\to\research_project"
   python src/models/train_classifier.py --fast
   ```
3. **Evaluate the checkpoint**
   ```powershell
   $env:PYTHONPATH="D:\path\to\research_project"
   python src/models/evaluate.py --model models/emotion_bert_small
   ```
4. **Run the chatbot API**
   ```powershell
   $env:PYTHONPATH="D:\path\to\research_project"
   python src/chatbot/app.py --model models/emotion_bert_small
   ```
   Hit `POST /chat` with `{"text": "your message"}` to receive an emotion-aware response.

## Dataset
- `data/emotion_chatbot_samples.csv` contains 56 curated utterances spanning 8 emotion classes (`joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `love`, `neutral`).
- Multi-turn threads are grouped via `conversation_id`/`turn` so that future context-aware models can track emotion transitions.
- Columns include both the user text (`text`) and the empathic response template (`bot_response`) which can seed the response generator module.

## Training & Evaluation Scripts
- `src/utils/dataset.py` tokenizes the CSV, creates train/validation/test splits, and returns a deterministic `label_map`.
- `src/models/train_classifier.py` fine-tunes `bert-base-uncased` and saves both model + tokenizer into `models/emotion_bert_small` (override via `--output`).
- `src/models/evaluate.py` loads any saved checkpoint and reports macro precision/recall/F1 on the held-out split using the same preprocessing pipeline.

## Chatbot Prototype
- `src/chatbot/response_selector.py` now reads the label mapping from the checkpoint, ensuring the Flask app automatically adapts to any emotion set learned during training.
- Run `src/chatbot/app.py` to start a simple Flask server with `/ping` and `/chat` endpoints; point it to the freshly fine-tuned directory via `--model`.
