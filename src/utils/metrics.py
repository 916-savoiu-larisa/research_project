from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}
