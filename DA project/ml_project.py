# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim


def load():
    batch_size = 64
    test_batch_size = 64
    img_size = 32

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mnist_transformations = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    svhn_transformations = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    svhn_train = datasets.SVHN("../data", download=True,
                               transform=svhn_transformations, split='train')
    svhn_test = datasets.SVHN("../data", download=True,
                              transform=svhn_transformations, split='test')
    mnist = datasets.MNIST("../data", train=False,
                           download=True, transform=mnist_transformations)

    svhn_train_loader = torch.utils.data.DataLoader(dataset=svhn_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    svhn_test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
                                                   batch_size=test_batch_size,
                                                   shuffle=True)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=test_batch_size,
                                               shuffle=True)

    return svhn_train_loader, svhn_test_loader, mnist_loader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(50, 100, kernel_size=5)
        self.batch_norm3 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(4 * 4 * 80, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.batch_norm1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.batch_norm2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.batch_norm3(x)
        x = x.view(-1, 4 * 4 * 100)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


epochs = 10
learning_rate = 0.001
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
svhn_train_loader, svhn_test_loader, mnist_loader = load()

max_acc = {"mnist": 0, "svhn_train": 0, "svhn_test": 0}
for epoch in range(1, epochs + 1):
    train(model, device, svhn_train_loader, optimizer)
    mnist_acc = test(model, device, mnist_loader)
    svhn_test_acc = test(model, device, svhn_test_loader)
    svhn_train_acc = test(model, device, svhn_train_loader)
    if mnist_acc > max_acc["mnist"]:
        max_acc["mnist"] = mnist_acc
    if svhn_test_acc > max_acc["svhn_test"]:
        max_acc["svhn_test"] = svhn_test_acc
    if svhn_train_acc > max_acc["svhn_train"]:
        max_acc["svhn_train"] = svhn_train_acc
print(max_acc["mnist"])
