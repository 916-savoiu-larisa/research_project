import random
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

FALLBACK_CLASS_TO_EMOTION = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'surprise', 5: 'neutral'}

# Enhanced keyword hints with more positive phrases - ORDER MATTERS: positive emotions checked first
KEYWORD_HINTS = [
    ('joy', ['happy', 'glad', 'excited', 'great day', 'smiling', 'proud', 'celebrate', 'promotion', 'over the moon', 
             'ecstatic', 'care package', 'feels possible', 'sun is out', 'everything feels', 'aced', 'fantastic',
             'wonderful', 'amazing', 'brilliant', 'thrilled', 'delighted', 'grateful', 'blessed', 'fortunate',
             'praised', 'made my whole week', 'made my week', 'whole week', 'outstanding', 'recognition',
             'achievement', 'success', 'accomplished', 'milestone', 'relieved', 'finished', 'completed',
             'surprised me with', 'surprise', 'good news', 'positive', 'great', 'excellent', 'perfect']),
    ('love', ['love', 'cherish', 'affection', 'partner', 'hug', 'gratitude', 'heartwarming', 'community support',
              'warmth', 'connected', 'appreciate', 'adore', 'treasure', 'full of love', 'overflowing',
              'filled my heart', 'supported', 'rallied around', 'thoughtful gesture']),
    ('sadness', ['sad', 'down', 'unhappy', 'gloomy', 'depressed', 'cry', 'crying', 'lonely', 'hopeless', 'lost', 'miss',
                 'rejected', 'invisible', 'gloom', 'weigh', 'hurt', 'discouraging', 'losing hope', 'can\'t stop crying']),
    ('anger', ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'frustrated', 'hate', 'can\'t stand', 'unfairly',
               'rage', 'infuriating', 'furious', 'making me mad', 'stuck in traffic', 'ignoring', 'interrupting']),
    ('fear', ['worried', 'anxious', 'scared', 'terrified', 'nervous', 'afraid', 'shaking', 'surgery', 'layoffs',
              'uncertainty', 'unsettling', 'thought of', 'has me shaking', 'worry about']),
    ('surprise', ['surprised', 'shocked', 'unexpected', 'wow', 'can\'t believe', 'surprised me', 'stunned',
                  'announced', 'learned', 'out of nowhere', 'came back negative']),
    ('disgust', ['disgust', 'gross', 'grossed out', 'nasty', 'sickened', 'unpleasant', 'revolting', 'feel sick',
                 'smell awful', 'unbearable', 'hateful comments', 'spoil'])
]


class ResponseSelector:
    def __init__(self, model_path, csv_path='data/emotion_chatbot_samples.csv'):
        # load classifier and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        self.class_to_emotion = self._extract_label_mapping()
        
        # Load CSV responses for more accurate matching
        self.csv_responses = None
        # Resolve CSV path - try relative to current dir, then relative to model path
        if not os.path.isabs(csv_path):
            # Try relative to current working directory
            if not os.path.exists(csv_path):
                # Try relative to model directory
                model_dir = os.path.dirname(os.path.abspath(model_path))
                csv_path_alt = os.path.join(model_dir, '..', csv_path)
                csv_path_alt = os.path.normpath(csv_path_alt)
                if os.path.exists(csv_path_alt):
                    csv_path = csv_path_alt
                    print(f"[ResponseSelector] Found CSV at: {csv_path}")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"[ResponseSelector] Reading CSV from: {os.path.abspath(csv_path)}")
                # Filter: only user rows (speaker='user') and non-empty bot_response
                df_filtered = df[(df['speaker'] == 'user') & (df['bot_response'].notna()) & (df['bot_response'] != '')]
                print(f"[ResponseSelector] Filtered to {len(df_filtered)} valid user rows with responses")
                if len(df_filtered) > 0:
                    self.csv_responses = df_filtered.groupby('emotion')['bot_response'].apply(list).to_dict()
                    # Convert to lowercase keys for matching
                    self.csv_responses = {k.lower(): v for k, v in self.csv_responses.items()}
                    total_responses = sum(len(v) for v in self.csv_responses.values())
                    print(f"[ResponseSelector] ✓ Loaded {total_responses} responses from CSV for {len(self.csv_responses)} emotions")
                    print(f"[ResponseSelector] Available emotions: {sorted(self.csv_responses.keys())}")
                else:
                    print(f"[ResponseSelector] ⚠ Warning: No valid responses found in CSV (all empty or bot rows)")
            except Exception as e:
                print(f"[ResponseSelector] ⚠ Warning: Could not load CSV responses: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[ResponseSelector] ⚠ Warning: CSV file not found at: {csv_path}")
            print(f"[ResponseSelector] Current working directory: {os.getcwd()}")
        
        # semantic ranking model (optional)
        try:
            self.semantic = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.semantic = None
        
        # Confidence threshold - if model confidence is below this, use keyword fallback
        self.confidence_threshold = 0.3

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
        """Enhanced keyword matching with priority for positive emotions"""
        lowered = text.lower()
        
        # Check for positive emotions FIRST to avoid false negatives
        # This is critical - positive phrases can be misclassified as negative
        positive_emotions = ['joy', 'love']
        for emotion_name, keywords in KEYWORD_HINTS:
            if emotion_name in positive_emotions:
                for keyword in keywords:
                    if keyword in lowered:
                        print(f"[ResponseSelector] Keyword match: '{keyword}' -> {emotion_name}")
                        return emotion_name
        
        # Then check other emotions (negative ones)
        for emotion_name, keywords in KEYWORD_HINTS:
            if emotion_name not in positive_emotions:
                for keyword in keywords:
                    if keyword in lowered:
                        print(f"[ResponseSelector] Keyword match: '{keyword}' -> {emotion_name}")
                        return emotion_name
        
        return None

    def predict_emotion(self, text):
        # ALWAYS try keyword override FIRST (especially for positive emotions)
        # This prevents false negatives where positive text is misclassified as negative
        hint = self._keyword_override(text)
        if hint:
            print(f"[ResponseSelector] Using keyword override: {hint}")
            return hint
        
        # Use model prediction with confidence check
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = float(probs.max().item())
            pred = int(logits.argmax(dim=-1).item())
            pred_emotion = self.class_to_emotion.get(pred, 'neutral')
        
        print(f"[ResponseSelector] Model prediction: {pred_emotion} (confidence: {confidence:.3f})")
        
        # If confidence is low OR if model predicts negative emotion but keywords suggest positive, use keyword fallback
        if confidence < self.confidence_threshold:
            hint = self._keyword_override(text)
            if hint:
                print(f"[ResponseSelector] Low confidence ({confidence:.3f}), using keyword override: {hint}")
                return hint
        
        # If model predicts negative emotion but we have positive keywords, double-check
        negative_emotions = ['anger', 'sadness', 'fear', 'disgust']
        if pred_emotion in negative_emotions:
            hint = self._keyword_override(text)
            if hint and hint in ['joy', 'love']:
                print(f"[ResponseSelector] Model predicted {pred_emotion} but keywords suggest {hint}, using {hint}")
                return hint
        
        return pred_emotion

    def select_response(self, text):
        emotion = self.predict_emotion(text)
        emotion_lower = emotion.lower()
        print(f"[ResponseSelector] Detected emotion: {emotion} (lowercase: {emotion_lower})")
        
        # Prefer CSV responses if available (more accurate)
        if self.csv_responses:
            print(f"[ResponseSelector] CSV responses available. Looking for: {emotion_lower}")
            # Try exact match first (already lowercase)
            if emotion_lower in self.csv_responses:
                responses = self.csv_responses[emotion_lower]
                if responses:
                    print(f"[ResponseSelector] ✓ Using CSV response for {emotion} ({len(responses)} options available)")
                    return random.choice(responses)
            else:
                print(f"[ResponseSelector] ⚠ Emotion '{emotion_lower}' not found in CSV. Available: {sorted(self.csv_responses.keys())}")
        else:
            print(f"[ResponseSelector] ⚠ No CSV responses loaded, using templates")
        
        # Fallback to templates if CSV not available
        TEMPLATES = {
            'joy': ["That's wonderful to hear! Tell me more.", 'So happy for you! 😊', 
                    "Sounds like a perfect moment—anything you'd like to channel that energy into?"],
            'sadness': ["I'm sorry you're feeling that way. Do you want to talk about it?", 
                       "That sounds tough; I'm here to listen."],
            'anger': ["I can hear you're upset. What's on your mind?", 
                     'That sounds frustrating. Do you want to vent?'],
            'fear': ['That sounds worrying — want to share more details?', 
                    'I understand, that can be scary.'],
            'surprise': ["Oh — that's surprising! What happened next?", 'Wow — tell me more!'],
            'neutral': ['I see. Can you tell me more?', 'Okay — what would you like to talk about?'],
            'disgust': ["That's definitely unpleasant — want to figure out a fix together?", 
                       'Totally fair to feel grossed out. Would taking action help?'],
            'love': ['That sounds so full of warmth—how might you share it?', 
                    'I love how connected you feel—want to celebrate it somehow?']
        }
        templates = TEMPLATES.get(emotion, TEMPLATES['neutral'])
        return random.choice(templates)
