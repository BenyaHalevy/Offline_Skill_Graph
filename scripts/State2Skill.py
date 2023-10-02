import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
random.seed(5)

root_path = Path(__file__).absolute().parent.parent

df = pd.read_csv(f"{root_path}\\Panda_Robot_Sim\\scripts\data_collection\datasets\classifier\merged\merged.csv")
df.head()

df['skill'] = df['skill'].astype('category')
encode_map = {
    'S11-S12': 0,
    'S12-S6':  1,
    'S12-S13': 2,
    'S13-S11': 3,
    'S10-S11': 4,
    'S6-S11':  5,
}

df['skill'].replace(encode_map, inplace=True)

X = df.iloc[:, 0:-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()


from sklearn.model_selection import train_test_split
x, x_val, y, y_val = train_test_split(X, Y, test_size=0.8, random_state=42)

x_train = x.reshape(-1, x.shape[1]).astype('float32')
y_train = y

x_val = x_val.reshape(-1, x_val.shape[1]).astype('float32')
y_val = y_val

x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val)

from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(x).type(torch.FloatTensor)
        self.y=torch.from_numpy(y).type(torch.LongTensor)
        self.len=self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
data_set=Data()
trainloader=DataLoader(dataset=data_set, batch_size=1028)


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


model = Net(10, 256, 256, 6)
# model_drop = Net(10, 128, 128, 4, p=0.2)
# model_drop

model.train()

optimizer_ofit = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
LOSS = {}
LOSS['training data no dropout'] = []
LOSS['validation data no dropout'] = []
LOSS['training data dropout'] = []
LOSS['validation data dropout'] = []
n_epochs = 9

for epoch in range(n_epochs):
    for x, y in trainloader:
        # make a prediction for both models
        yhat = model(data_set.x)
        # yhat_drop = model_drop(data_set.x)
        # calculate the lossf or both models
        loss = criterion(yhat, data_set.y)
        # loss_drop = criterion(yhat_drop, data_set.y)

        # store the loss for  both the training and validation  data for both models
        LOSS['training data no dropout'].append(loss.item())
        # LOSS['training data dropout'].append(loss_drop.item())
        # model_drop.eval()
        # model_drop.train()

        # clear gradient
        optimizer_ofit.zero_grad()
        # optimizer_drop.zero_grad()
        # Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        # loss_drop.backward()
        # the step function on an Optimizer makes an update to its parameters
        optimizer_ofit.step()
        # optimizer_drop.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

z = model(x_val)
# z_dropout = model_drop(x_val)
_,yhat=torch.max(z.data,1)
# _,yhat_dropout=torch.max(z_dropout.data,1)

# Making the Confusion Matrix
eval_matrix = (pd.crosstab(y_val, yhat))
print(eval_matrix)

# Making the Confusion Matrix
# eval_matrix_dropout = (pd.crosstab(y_val, yhat_dropout))
# print(eval_matrix_dropout)

print('y_val len: ', y_val.shape[0])
print('matrix len: ', len(eval_matrix))

for i in range(1,len(eval_matrix)):
    eval_score_1 = eval_matrix[i][i]
eval_score_1 = eval_score_1/y_val.shape[0]
print(eval_score_1)
eval_score_2 = eval_matrix[0][0]
for i in range(1,len(eval_matrix)):
    eval_score_2 = eval_score_2 + eval_matrix[i][i]+eval_matrix[i-1][i]+eval_matrix[i][i-1]
eval_score_2 = eval_score_2/y_val.shape[0]
print(eval_score_2)

# print((eval_matrix_dropout[0][0]+eval_matrix_dropout[1][1]+eval_matrix_dropout[2][2])/y_val.shape[0])

torch.save(model.state_dict(), 'State2Skill.pth')

