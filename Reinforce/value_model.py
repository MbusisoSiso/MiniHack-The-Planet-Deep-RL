import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, s_size=4, h_size=16, learning_rate=0.001):
        super(ValueNetwork, self).__init__()
        self.linear_1 = nn.Linear(s_size, h_size)
        self.linear_2 = nn.Linear(h_size, h_size)
        self.linear_3 = nn.Linear(h_size, 1)
        self.adam = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.reshape(x, (1,x.shape[0]))
        f = F.relu(self.linear_1(x))
        f = F.relu(self.linear_2(f))
        f = self.linear_3(f)
        return f