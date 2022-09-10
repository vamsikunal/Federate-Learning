from torchvision import datasets, transforms

class DatasetLoad():
    def __init__(self):
        self.tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),])

    def load_mnist(self):
        train_set = datasets.MNIST(
            "~/.pytorch/MNIST_data/", train=True, download=True, transform=self.transform)
        test_set = datasets.MNIST(
            "~/.pytorch/MNIST_data/", train=False, download=True, transform=self.transform)
        return (train_set, test_set);

    def load_cifar10(self):
        train_set = datasets.CIFAR10(
            "~/.pytorch/CIFAR10_data/", train=True, download=True, transform=self.transform)
        test_set = datasets.CIFAR10(
            "~/.pytorch/CIFAR10_data/", train=False, download=True, transform=self.transform)
        return (train_set, test_set);
    
    def load_fashion_mnist(self):
        train_set = datasets.FashionMNIST(
            "~/.pytorch/FashionMNIST_data/", train=True, download=True, transform=self.transform)
        test_set = datasets.FashionMNIST(
            "~/.pytorch/FashionMNIST_data/", train=False, download=True, transform=self.transform)
        return (train_set, test_set);





