
import numpy as np
import torch
import torchtext
from torch.utils import data
import ipdb
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, device):

        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # Forward propagate LSTM
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        print("\t\t Device:", device, "In Model: input size", x.size(), "output size", out.size())
        return out

def cal_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss

def optimizer():
    return torch.optim.Adam

def init_model(config_dic, num_classes):
    return RNN(config_dic['model_params']['input_size'], 
               config_dic['model_params']['hidden_size'],
               config_dic['model_params']['num_layers'],
               num_classes)


