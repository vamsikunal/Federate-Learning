from dataset import DatasetLoad
import torch
import syft as sy  # import the syft library



hook = sy.TorchHook(torch)  # attach the pytorch hook
kunal = sy.VirtualWorker(hook, id="kunal")  #  remote worker 
aayush = sy.VirtualWorker(hook, id="aayush")  
saatvik = sy.VirtualWorker(hook, id="saatvik")

def locked_data(df):
    federated_train_loader = sy.FederatedDataLoader(
        df[0].federate((kunal, aayush, saatvik)), batch_size=128, shuffle=True) # the federate() method splits the data within the workers

    test_loader = torch.utils.data.DataLoader(
        df[1], batch_size=128, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        df[0], batch_size=128, shuffle=True)
    return (federated_train_loader, test_loader)