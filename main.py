from model import MyModel
from model_train import TrainMyModel
from model_eval import ModelEvaluation
from pysyft import *
from dataset import DatasetLoad


df = DatasetLoad().load_mnist()
model = MyModel()
federated_train_loader, test_loader = locked_data(df)
n_epoch = 3


train_model = TrainMyModel(model, federated_train_loader, n_epoch)
model = train_model.train()

model_eval = ModelEvaluation(model, test_loader)
model_eval.evluation()
print(model_eval.accuracy())
