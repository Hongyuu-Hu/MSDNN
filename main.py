import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from network import Sentiment # Assuming MultiScaleFeatureModel is not used
from units import evaluate
import seed_set

seed =seed_set.seed
# 数据预处理
# 加载数据
df = pd.read_csv('demo.csv')
texts = df['review'].values
labels = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment']]

# 划分训练集和测试集
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenization and Vocabulary Building
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(texts_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    return vocab(tokenizer(x))

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return text_pipeline(self.texts[idx]), self.labels[idx]

train_data = TextDataset(texts_train, labels_train)
test_data = TextDataset(texts_test, labels_test)

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        text_list.append(Tensor(_text).to(torch.int64))
        label_list.append(_label)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=3.0)
    return text_list, label_list

train_dataloader = DataLoader(train_data, batch_size=1000, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False, collate_fn=collate_batch)

# 模型初始化
vocab_size = len(vocab)
embed_dim = 32
num_filters = 100
filter_sizes = [2, 3, 4]
num_classes = len(set(labels))
model = Sentiment(vocab_size, embed_dim, num_filters, filter_sizes, num_classes)
model = model.to('cuda')  # 将模型移到CUDA设备上
classification_criterion = nn.CrossEntropyLoss()
classification_criterion = classification_criterion.to('cuda')  # 将损失函数移到CUDA设备上
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(dataloader, model, classification_criterion, optimizer):
    model.train()
    total_loss = 0
    for text, labels in dataloader:
        optimizer.zero_grad()
        features, class_scores = model(text.t().to('cuda'))
        classification_loss = classification_criterion(class_scores, labels.to('cuda'))
        loss = classification_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练过程\
best_accuracy=0
num_epochs = 1000
Loss_save = []
for epoch in range(num_epochs):
    loss = train(train_dataloader, model, classification_criterion, optimizer)
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(test_dataloader, model, classification_criterion)
    # Update best accuracy and epoch if current accuracy is better
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_Precision=test_precision
        best_Recall  = test_recall
        best_F1 = test_f1
        best_epoch = epoch + 1

    Loss_save.append(loss)
    print(f'Epoch {epoch + 1}, Training Loss: {loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')

with open('loss.txt', 'w') as file:
    for loss in Loss_save:
        file.write(str(loss) + '\n')
# Print the best result
print(f'\nBest Accuracy: {best_accuracy:.3f},Best Precision: {best_Precision:.3f} ,Best Recall: {best_Recall:.3f} ,Best F1: {best_F1:.3f}  at Epoch {best_epoch}')
with open(f"result/result_p.txt", 'a', newline='', encoding='utf-8') as f:
    f.write(f'Seed: {seed},Best ACC: {best_accuracy:.3f}, Best PRE: {best_Precision:.3f} ,Best Recall: {best_Recall:.3f},Best F1: {best_F1:.3f}   at Epoch: {best_epoch}\n')