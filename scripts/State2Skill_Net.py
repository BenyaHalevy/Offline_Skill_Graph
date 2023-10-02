import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size, p=0):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden1)
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.linear3 = nn.Linear(n_hidden2, n_hidden2)
        nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        self.linear4 = nn.Linear(n_hidden2, out_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.drop(x)
        x = F.relu(self.linear2(x))
        x = self.drop(x)
        x = F.relu(self.linear3(x))
        x = self.drop(x)
        x = self.linear4(x)
        return x