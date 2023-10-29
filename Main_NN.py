import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from NeuralNetwork import ThreeLayerNN

def main():
    train_batch_size = 32
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), torch.flatten])), batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('MNIST/', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), torch.flatten])), batch_size=10000, shuffle=True)

    epochs = 3

    nn_model = ThreeLayerNN()

    learning_curve = {}
    iterations = 1

    for i in range(epochs):
        for step, (X_train, y_train) in enumerate(train_loader):
            y_train = y_train.reshape(1, y_train.shape[0])
            y_train = np.eye(10)[y_train].T.reshape(10, y_train.shape[1])

            curr_cost = nn_model.train(np.array(X_train), np.array(y_train))

            if iterations%100 == 0 or iterations == 1: 
                print(f'Epoch {i}, Step {step+1}, Cost = {curr_cost}')
                for idx, (X_test, y_test) in enumerate(test_loader):
                    learning_curve[iterations] = nn_model.test_error(np.array(X_test), np.array(y_test))
            iterations += 1

    accuracy = []
    for idx, (X_test, y_test) in enumerate(test_loader):
        accuracy.append(1 - nn_model.test_error(np.array(X_test), np.array(y_test)))

    print(np.mean(accuracy))

    # print(learning_curve)
    plt.plot(learning_curve.keys(), learning_curve.values())
    plt.savefig('Learning Curve Custom.png')

if __name__ == '__main__':
    main()