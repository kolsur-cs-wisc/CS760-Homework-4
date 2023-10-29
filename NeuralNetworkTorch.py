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
            nn.Softmax(dim=1)
        )
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            # nn.init.uniform_(param, -1, 1)
            nn.init.zeros_(param)
            print(name)

    def forward(self, x):
        return self.layers(x)
    
train_batch_size = 32
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=True, download=True, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(), torch.flatten])), batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=False, download=True, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(), torch.flatten])), batch_size=10000, shuffle=True)

def test_error(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    return 1 - correct

model = NeuralNetworkTorch()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

iterations = 1
epochs = 25
learning_curve = {}

print(f'Epoch {-1}, Test Error = {test_error(test_loader, model)}')
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
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        if iterations%100 == 0 or iterations == 1:
            learning_curve[iterations] = 1 - test_error(test_loader, model)
        iterations += 1
    print(f'Epoch {t}, Test Error = {test_error(test_loader, model)}')

plt.plot(learning_curve.keys(), learning_curve.values())
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Learning Curve Zero Weights (PyTorch Implementation)")
plt.savefig('Learning Curve PyTorch Zero Weights.png')