# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .CNN import CNN


class MyParaCNN(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(MyParaCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.dropout = nn.Dropout(config.dropout_embed)

        self.conv = CNN(config.in_channels, config.out_channels, embed_dim, config.kernel_sizes)

        self.lstm = nn.LSTM(config.out_channels, config.p_hidden_size, num_layers=config.p_num_layers,
                            dropout=config.p_dropout_rnn, bidirectional=True)

        self.output_linear = nn.Linear(config.p_hidden_size * 2, label_size)

    def forward(self, inputs, inputs_lengths):
        all_x = []

        for idx, x in enumerate(inputs):
            x = torch.transpose(x, 0, 1)
            x = self.embedding(x)
            x = self.dropout(x)

            x = self.conv(x)
            all_x.append(x)

        # 这个时候batch_size要不要增大到句子个数
        # 先试一试batch_size大于句子最大长度，设为150吧
        x = all_x[0].unsqueeze(1)
        x, _ = self.lstm(x)

        x = x.squeeze(1)
        x = torch.mean(x, 0).unsqueeze(0)
        x = self.output_linear(x)
        return x