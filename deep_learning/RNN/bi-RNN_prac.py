import pandas as pd
import torch
import chardet
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import TabularDataset, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import os

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=200)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

train_data, test_data = TabularDataset.splits(
        path='D:/ruin/data/test/', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

TEXT.build_vocab(train_data, min_freq=5, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

train_data, val_data = train_data.split(split_ratio=0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), sort=False,batch_size=64,
        shuffle=True, repeat=False)

class biRNN(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, output_dim, dropout, bidirectional):
        super(biRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          bidirectional=bidirectional,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.dropout(self.embed(x))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        logit = self.out(hidden)
        return logit

epochs = 10
input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 20
output_dim = 1
n_layers = 2 # for mutil layer rnn
bidirectional = True
dropout = 0.5
lr = 0.0001

## 참조 : https://jovian.ai/bhartikukreja2015/assign5-course-project-text-2/v/6?utm_source=embed