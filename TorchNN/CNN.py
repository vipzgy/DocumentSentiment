# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, embed_dim, kernel_size):
        super(CNN, self).__init__()

        Ks = [int(i) for i in kernel_size.split(',')]
        self.convs = [nn.Conv2d(in_channel, out_channel, (k, embed_dim)) for k in Ks]
        for conv in self.convs:
            conv = conv.cuda()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = [F.tanh(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)) for i in x]
        x = torch.cat(x, 2)
        x = torch.mean(x, 2)
        return x
