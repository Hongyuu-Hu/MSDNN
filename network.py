import torch
import torch.nn as nn
import torch.nn.functional as F

# 多尺度特征学习模型
class MultiScaleFeatureModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super(MultiScaleFeatureModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        convoluted = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convoluted]
        cat = torch.cat(pooled, dim=1)
        return cat

# 情感分析模型
class Sentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout_rate=0.5):
        super(Sentiment, self).__init__()
        self.multi_scale_model = MultiScaleFeatureModel(vocab_size, embed_dim, num_filters, filter_sizes)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, text):
        features = self.multi_scale_model(text)
        features = self.dropout(features)
        class_scores = self.fc(features)
        return features, class_scores



class Sentiment_a(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_size, num_classes, dropout_rate=0.5):
        super(Sentiment_a, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=filter_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        convoluted = F.relu(self.conv(embedded))
        pooled = F.max_pool1d(convoluted, convoluted.shape[2]).squeeze(2)
        features = self.dropout(pooled)
        class_scores = self.fc(features)
        return features, class_scores
