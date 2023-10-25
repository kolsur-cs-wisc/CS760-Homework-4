import numpy as np
import torch
import torchvision
from NeuralNetwork import ThreeLayerNN
from sklearn.preprocessing import OneHotEncoder

def main():
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), torch.flatten])), batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), torch.flatten])), batch_size=100, shuffle=True)
    
    train_set = enumerate(train_loader)
    test_set = enumerate(test_loader)

    batch_idx, (X_train, y_train) = next(train_set)
    y_train = y_train.reshape(1, y_train.shape[0])
    y_train = np.eye(10)[y_train].T.reshape(10, y_train.shape[1])

    batch_idx, (X_test, y_test) = next(test_set)

    nn_model = ThreeLayerNN(np.array(X_train), np.array(y_train))
    nn_model.train()
    print(nn_model.test_error(np.array(X_test), np.array(y_test)))

if __name__ == '__main__':
    main()