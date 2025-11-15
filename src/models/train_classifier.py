#!/usr/bin/env python3
"""Minimal training script using Hugging Face transformers and datasets.
This script supports a --fast mode for CI/testing that uses a tiny dataset and 1 epoch.
"""
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.utils.metrics import compute_metrics
from src.utils.dataset import load_local_csv_and_prepare

MODEL_NAME = 'bert-base-uncased'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # load and preprocess
    ds = load_local_csv_and_prepare(args.data, tokenizer, max_length=128, label_map=None)
    if args.fast:
        # take tiny subset for CI speed
        for split in ds:
            ds[split] = ds[split].select(range(min(32, len(ds[split]))))
        args.epochs = 1

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=args.num_labels)
    training_args = TrainingArguments(
        output_dir=args.output,
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )

    # tokenize done already in dataset helper; ensure torch format
    ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/goemotions_subset.csv')
    parser.add_argument('--output', type=str, default='models/emotion_bert_small')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_labels', type=int, default=6)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    main(args)
