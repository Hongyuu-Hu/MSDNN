import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(dataloader, model, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for text, labels in dataloader:
            text, labels = text.t().to('cuda'), labels.to('cuda')
            features, class_scores = model(text)
            loss = criterion(class_scores, labels)
            total_loss += loss.item()

            predictions = class_scores.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return total_loss / len(dataloader), accuracy, precision, recall, f1
