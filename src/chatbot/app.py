#!/usr/bin/env python3
from flask import Flask, request, jsonify
import argparse, os, random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.chatbot.response_selector import ResponseSelector

app = Flask(__name__)
selector = None

@app.route('/ping')
def ping():
    return 'pong'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error':'missing text field'}), 400
    text = data['text']
    resp = selector.select_response(text)
    return jsonify({'response': resp})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/emotion_bert_small')
    args = parser.parse_args()
    global selector
    selector = ResponseSelector(args.model)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))

if __name__ == '__main__':
    main()
