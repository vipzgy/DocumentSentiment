# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .LSTM import LSTM


class MyParaLSTM(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_idx, label_size, embedding=None):
        super(MyParaLSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(embed_size, embed_dim, padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.dropout = nn.Dropout(config.dropout_embed)

        self.s_lstm = LSTM(embed_dim, config.s_hidden_size, config.s_num_layers, config.s_dropout_rnn)

        self.p_lstm = nn.LSTM(config.s_hidden_size * 2, config.p_hidden_size, num_layers=config.p_num_layers,
                              dropout=config.p_dropout_rnn, bidirectional=True)

        self.output_linear = nn.Linear(config.p_hidden_size * 2, label_size)

    def forward(self, inputs, inputs_lengths):
        all_x = []

        for idx, x in enumerate(inputs):

            x = self.embedding(x)
            x = self.dropout(x)

            x = self.s_lstm(x, inputs_lengths[idx])
            all_x.append(x)

        # 这个时候batch_size要不要增大到句子个数
        # 先试一试batch_size大于句子最大长度，设为150吧
        x = all_x[0].unsqueeze(1)
        x, _ = self.p_lstm(x)

        x = x.squeeze(1)
        x = torch.mean(x, 0).unsqueeze(0)
        x = self.output_linear(x)
        return x