import pandas as pd
from datasets import Dataset, DatasetDict


def load_local_csv_and_prepare(path, tokenizer, max_length=128, label_map=None):
    df = pd.read_csv(path)
    print(f"[dataset] Loaded {len(df)} rows from {path}")
    # Expect columns 'text' and 'label'
    if label_map is None:
        unique_labels = sorted(df['label'].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"[dataset] Generated label map: {label_map}")
    df['label'] = df['label'].map(label_map)
    if df['label'].isnull().any():
        missing = df[df['label'].isnull()]
        raise ValueError(f'Label mapping missing for values: {missing}')
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
    return ds, label_map
