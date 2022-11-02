import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, s_size, h_size, a_size, learning_rate=0.001):
        super(PolicyNetwork, self).__init__()
        self.linear_1 = nn.Linear(s_size, h_size)
        self.linear_2 = nn.Linear(h_size, h_size)
        self.linear_3 = nn.Linear(h_size, a_size)
        self.loss=nn.CrossEntropyLoss()
        self.adam = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.reshape(x, (1,x.shape[0]))
        x = torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12, out=None)
        f = F.relu(self.linear_1(x))
        f = F.relu(self.linear_2(f))
        f = F.softmax(self.linear_3(f), dim=1)
        return f
