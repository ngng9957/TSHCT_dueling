#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(RNNAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.rnn_hidden_dim = 128

        self.fc1 = nn.Linear(self.num_inputs, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.fc1_val = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc1_adv = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.fc2_val = nn.Linear(self.rnn_hidden_dim, 1)
        self.fc2_adv = nn.Linear(self.rnn_hidden_dim, self.num_outputs)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        val = F.relu(self.fc1_val(h))
        adv = F.relu(self.fc1_adv(h))

        val = self.fc2_val(val)
        adv = self.fc2_adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)

        q = val + adv - advAverage
        return q, h

    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)