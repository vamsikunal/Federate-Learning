from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

def load_mnist():
    train_set = datasets.MNIST(
        "~/.pytorch/MNIST_data/", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(
        "~/.pytorch/MNIST_data/", train=False, download=True, transform=transform)
    return (train_set, test_set);

