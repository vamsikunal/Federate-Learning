from pysyft import *
from model import MyModel
import torch.optim as optim
import torch.nn.functional as F

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.05)
federated_train_loader, test_loader = locked_data()
n_epoch = 3
for epoch in range(n_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location) # send the model to the client device where the data is present
        optimizer.zero_grad()         # training the model
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        model.get() # get back the improved model
        if batch_idx % 100 == 0: 
            loss = loss.get() # get back the loss
            print('Training Epoch: {:2d} [{:5d}/{:5d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * 128,
                len(federated_train_loader) * 128,
                100. * batch_idx / len(federated_train_loader), loss.item()))