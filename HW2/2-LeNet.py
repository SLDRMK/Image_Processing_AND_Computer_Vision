import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

# print(torch.cuda.is_available())

batch_size = 256
lr = 0.5
num_epochs = 20

lenet = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(in_features=120, out_features=84),
    nn.Sigmoid(),
    nn.Linear(in_features=84, out_features=10)
)

class MNIST():
    def __init__(self, batch_size):
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.train_data = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=False,
            transform=transforms
            )
        self.test_data = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=False,
            transform=transforms
            )

        self.train_iter = DataLoader(
            dataset=self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)
        self.test_iter = DataLoader(
            dataset=self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

class Accumulator():
    def __init__(self, num):
        self.num = num
        self.data = [0 for x in range(num)]

    def add(self, *args):
        for index in range(self.num):
            self.data[index] += args[index]
    
    def __len__(self):
        return self.num
    
    def __get_item__(self, index):
        return self.data[index]

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.max(y_hat, axis=1).indices
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    
    metric = Accumulator(num=2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric.__get_item__(0) / metric.__get_item__(1)

def train_net(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            
            train_l = metric.__get_item__(0) / metric.__get_item__(2)
            train_acc = metric.__get_item__(1) / metric.__get_item__(2)
            
        test_acc = evaluate_accuracy(net, test_iter)
        print("train_epoch {}: test_acc {},train_acc {},train_l {}".format(epoch, test_acc, train_acc, train_l))

if __name__ == "__main__":
    mnist = MNIST(batch_size=batch_size)
    train_net(
        lenet,
        train_iter=mnist.train_iter,
        test_iter=mnist.test_iter,
        num_epochs=num_epochs,
        lr=lr,
        device="cuda:0")