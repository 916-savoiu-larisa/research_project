import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np

def default_label_map(label):
    # example mapping placeholder (user should adapt to dataset labels)
    return label

def load_local_csv_and_prepare(path, tokenizer, max_length=128, label_map=None):
    df = pd.read_csv(path)
    # Expect columns 'text' and 'label'
    if label_map is None:
        label_map = lambda x: x
    df['label'] = df['label'].apply(label_map)
    # simple split
    n = len(df)
    train = df.sample(frac=0.7, random_state=42)
    rest = df.drop(train.index)
    val = rest.sample(frac=0.5, random_state=42)
    test = rest.drop(val.index)
    def tokenize_batch(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    ds = DatasetDict({
        'train': Dataset.from_pandas(train.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val.reset_index(drop=True)),
        'test': Dataset.from_pandas(test.reset_index(drop=True))
    })
    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.rename_column('label', 'labels')
    return ds
