# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers, dropout_rnn):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout_rnn, batch_first=True, bidirectional=True)

    def forward(self, x, x_lengths):
        x = pack_padded_sequence(x, x_lengths)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x)
        # x = torch.transpose(x, 0, 2)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = torch.mean(x, 0)

        return x
