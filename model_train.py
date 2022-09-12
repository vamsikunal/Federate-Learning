from pysyft import *
from model import MyModel
import torch.optim as optim
import torch.nn.functional as F
import progressbar


# model = MyModel()
# optimizer = optim.SGD(model.parameters(), lr=0.05)
# federated_train_loader, test_loader = locked_data()
# n_epoch = 3

class TrainMyModel:

    def __init__(self, model, federated_train_loader, n_epoch):
        self.model = model
        self.federated_train_loader = federated_train_loader
        self.optimizer = model.optimizer
        self.n_epoch = n_epoch
        self.loss_list = []
        
    def train(self):
        for self.epoch in range(self.n_epoch):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.federated_train_loader):
                self.model.send(data.location) # send the model to the client device where the data is present
                self.optimizer.zero_grad()         # training the model
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                self.model.get() 
            loss = loss.get()# get back the improved model
            self.loss_list += [loss.item()]
        return self.model
    
    def get_loss_list(self):
        return self.loss_list


