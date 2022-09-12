import torch
import torch.nn.functional as F


class ModelEvaluation:

    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.correct = 0

    def evluation(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
                self.correct += pred.eq(target.view_as(pred)).sum().item()
    
    def accuracy(self):
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * self.correct / len(self.test_loader.dataset)
        return accuracy