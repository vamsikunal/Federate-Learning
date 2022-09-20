from pysyft import *
from model import MyModel
import torch.optim as optim
import torch.nn.functional as F
import progressbar

class TrainMyModel:

    def __init__(self, model, federated_train_loader, n_epoch):
        self.model = model
        self.federated_train_loader = federated_train_loader
        self.optimizer = model.optimizer
        self.n_epoch = n_epoch
        self.loss_list = []
        
    def train(self): 
        for self.epoch in range(self.n_epoch):
            print("Epoch: ", self.epoch, "\n")
            self.model.train()
            with progressbar.ProgressBar(max_value=batch_idx) as bar:
                for batch_idx, (data, target) in enumerate(self.federated_train_loader):
                    self.model.send(data.location) # send the model to the client device where the data is present
                    self.optimizer.zero_grad()         # training the model
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.optimizer.step()
                    self.model.get() 
                    bar.update(batch_idx)
            loss = loss.get()# get back the improved model
            self.loss_list += [loss.item()]
        return self.model
    
    def get_loss_list(self):
        return self.loss_list


