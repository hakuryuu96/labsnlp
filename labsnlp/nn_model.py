import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyLSTMClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim): 
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        o_n = self.linear(hn)
        classes = torch.sigmoid(o_n)[0]
        return classes


if __name__ == "__main__":
    model = MyLSTMClassifier(20, 30, 40)
    x = torch.ones((20, 30))

    print(model(x))