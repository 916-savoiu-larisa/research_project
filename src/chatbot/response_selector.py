import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch

FALLBACK_CLASS_TO_EMOTION = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'surprise', 5: 'neutral'}

TEMPLATES = {
    'joy': ["That's wonderful to hear! Tell me more.", 'So happy for you! 😊'],
    'sadness': ["I'm sorry you're feeling that way. Do you want to talk about it?", "That sounds tough; I'm here to listen."],
    'anger': ["I can hear you're upset. What's on your mind?", 'That sounds frustrating. Do you want to vent?'],
    'fear': ['That sounds worrying — want to share more details?', 'I understand, that can be scary.'],
    'surprise': ["Oh — that's surprising! What happened next?", 'Wow — tell me more!'],
    'neutral': ['I see. Can you tell me more?', 'Okay — what would you like to talk about?'],
    'disgust': ["That's definitely unpleasant — want to figure out a fix together?", 'Totally fair to feel grossed out. Would taking action help?'],
    'love': ['That sounds so full of warmth—how might you share it?', 'I love how connected you feel—want to celebrate it somehow?']
}

KEYWORD_HINTS = [
    ('joy', ['happy', 'glad', 'excited', 'great day', 'smiling', 'proud', 'celebrate', 'promotion', 'over the moon', 'ecstatic', 'care package']),
    ('love', ['love', 'cherish', 'affection', 'partner', 'hug', 'gratitude', 'heartwarming', 'community support']),
    ('sadness', ['sad', 'down', 'unhappy', 'gloomy', 'depressed', 'cry', 'lonely', 'hopeless']),
    ('anger', ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'frustrated']),
    ('fear', ['worried', 'anxious', 'scared', 'terrified', 'nervous', 'afraid']),
    ('surprise', ['surprised', 'shocked', 'unexpected', 'wow', 'can\'t believe']),
    ('disgust', ['disgust', 'gross', 'nasty', 'sickened'])
]


class ResponseSelector:
    def __init__(self, model_path):
        # load classifier and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.class_to_emotion = self._extract_label_mapping()
        # semantic ranking model (optional)
        try:
            self.semantic = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.semantic = None

    def _extract_label_mapping(self):
        id2label = getattr(self.model.config, 'id2label', None) or {}
        normalized = {}
        for key, value in id2label.items():
            try:
                idx = int(key)
            except (ValueError, TypeError):
                idx = key
            normalized[idx] = value.lower()
        if not normalized:
            normalized = FALLBACK_CLASS_TO_EMOTION
        return normalized

    def _keyword_override(self, text):
        lowered = text.lower()
        for emotion, keywords in KEYWORD_HINTS:
            if any(keyword in lowered for keyword in keywords):
                return emotion
        return None

    def predict_emotion(self, text):
        hint = self._keyword_override(text)
        if hint:
            return hint
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = int(logits.argmax(dim=-1).item())
        return self.class_to_emotion.get(pred, 'neutral')

    def select_response(self, text):
        emotion = self.predict_emotion(text)
        templates = TEMPLATES.get(emotion, TEMPLATES['neutral'])
        return random.choice(templates)
