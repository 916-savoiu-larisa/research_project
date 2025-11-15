import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch

CLASS_TO_EMOTION = {0:'joy',1:'sadness',2:'anger',3:'fear',4:'surprise',5:'neutral'}

TEMPLATES = {
    'joy': ['That\'s wonderful to hear! Tell me more.', 'So happy for you! 😊'],
    'sadness': ['I\'m sorry you\'re feeling that way. Do you want to talk about it?', 'That sounds tough; I\'m here to listen.'],
    'anger': ['I can hear you\'re upset. What\'s on your mind?', 'That sounds frustrating. Do you want to vent?'],
    'fear': ['That sounds worrying — want to share more details?', 'I understand, that can be scary.'],
    'surprise': ['Oh — that\'s surprising! What happened next?', 'Wow — tell me more!'],
    'neutral': ['I see. Can you tell me more?', 'Okay — what would you like to talk about?']
}

class ResponseSelector:
    def __init__(self, model_path):
        # load classifier and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # semantic ranking model (optional)
        try:
            self.semantic = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.semantic = None

    def predict_emotion(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = int(logits.argmax(dim=-1).item())
        return CLASS_TO_EMOTION.get(pred, 'neutral')

    def select_response(self, text):
        emotion = self.predict_emotion(text)
        templates = TEMPLATES.get(emotion, TEMPLATES['neutral'])
        return random.choice(templates)
