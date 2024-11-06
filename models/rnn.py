from torch import nn
import torch

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, tensor):