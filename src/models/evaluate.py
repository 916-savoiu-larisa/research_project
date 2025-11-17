#!/usr/bin/env python3
"""
Evaluate a fine-tuned classifier on the prepared dataset splits.
Usage:
    PYTHONPATH=. python src/models/evaluate.py --model models/emotion_bert_small --data data/emotion_chatbot_samples.csv
"""
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.utils.dataset import load_local_csv_and_prepare
from src.utils.metrics import compute_metrics


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    ds, _ = load_local_csv_and_prepare(args.data, tokenizer, max_length=128, label_map=None)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    eval_args = TrainingArguments(
        output_dir=args.output,
        per_device_eval_batch_size=16,
        eval_strategy='no',
        save_strategy='no',
        do_train=False,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=ds['test'],
        compute_metrics=compute_metrics
    )
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/emotion_bert_small')
    parser.add_argument('--data', type=str, default='data/emotion_chatbot_samples.csv')
    parser.add_argument('--output', type=str, default='models/eval_runs')
    args = parser.parse_args()
    main(args)

