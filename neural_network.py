import torch as tch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        tch.manual_seed(2020)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        self.out = nn.Linear(8, 5)
        self.final = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(0.1)  #Dropout Layer

    def forward(self, x):
        op = self.drop(x)  #Dropout para ingreso de capa 
        op = self.fc1(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout capa 1
        op = self.fc2(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout capa 2
        op = self.fc3(op)
        op = self.relu(op)
        op = self.drop(op) #Dropout capa 3
        op = self.out(op)
        y = self.final(op)
        return y