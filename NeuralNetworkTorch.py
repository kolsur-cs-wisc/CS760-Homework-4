import numpy as np
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt

class NeuralNetworkTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 300, bias=False),
            nn.Sigmoid(),
            nn.Linear(300, 10, bias=False),
            nn.Softmax()
        )
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param, -1, 1)
            # nn.init.zeros_(param)
            print(name)

    def forward(self, x):
        return self.layers(x)
    
train_batch_size = 32
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=True, download=True, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(), torch.flatten])), batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=False, download=True, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(), torch.flatten])), batch_size=10000, shuffle=True)

def test_error(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return 1 - correct

model = NeuralNetworkTorch()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

iterations = 1
epochs = 10
learning_curve = {}

for t in range(epochs):
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        if iterations%100 == 0 or iterations == 1:
            learning_curve[iterations] = test_error(test_loader, model, loss_fn)
        iterations += 1
    print(f'Test Error = {test_error(test_loader, model, loss_fn)}')

plt.plot(learning_curve.keys(), learning_curve.values())
plt.savefig('Learning Curve PyTorch.png')