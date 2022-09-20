from asyncio.windows_events import NULL
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 10)
        self.feature_extractor = None

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        self.feature_extractor = x
        return F.log_softmax(x, dim=1)
    
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.05)

    def get_output_base_model(self, x):
        pass