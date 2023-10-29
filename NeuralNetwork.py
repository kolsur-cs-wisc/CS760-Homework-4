import numpy as np
import pandas as pd

class ThreeLayerNN():
    def __init__(self, d = 784, k = 10, d1 = 300, lr = 0.03):
        self.lr = lr
        self.n_X = d
        self.n_y = k
        self.n_h = d1
        self.initialize_weights()

    def initialize_weights(self, method = 'range'):
        if method == 'zeros':
            self.W1 = np.zeros((self.n_h, self.n_X))
            self.W2 = np.zeros((self.n_y, self.n_h))
        else:
            self.W1 = np.random.uniform(-1, 1, (self.n_h, self.n_X))
            self.W2 = np.random.uniform(-1, 1, (self.n_y, self.n_h))

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def delta_sigmoid(self, value):
        return self.sigmoid(value) * (1 - self.sigmoid(value))

    def train(self, X_train, y_train):
        batch_cost = 0
        for X, y in zip(X_train, y_train.T):
            self.X = np.reshape(X, (1, self.n_X))
            self.y = np.reshape(y, (self.n_y, 1))
            self.m = self.X.shape[0]
            
            z1 = np.matmul(self.W1, self.X.T)
            a1 = self.sigmoid(z1)
            z2 = np.matmul(self.W2, a1)
            f = np.exp(z2) / np.sum(np.exp(z2), axis=0)

            batch_cost += self.loss_func(f, self.y)

            dz2 = f - self.y
            dW2 = (1/self.m) * np.matmul(dz2, a1.T)

            self.W2 = self.W2 - self.lr * dW2

            da1 = np.matmul(self.W2.T, dz2)
            dz1 = da1 * self.delta_sigmoid(z1)
            dW1 = (1./self.m) * np.matmul(dz1, self.X)

            self.W1 = self.W1 - self.lr * dW1

        return batch_cost

    def loss_func(self, y_pred, y_label):
        return -np.sum(np.multiply(y_label, np.log(y_pred)))
    
    def predict(self, X_test):
        h1 = self.sigmoid(np.matmul(self.W1, X_test.T))
        z2 = np.matmul(self.W2, h1)
        g = np.exp(z2) / np.sum(np.exp(z2), axis=0)
        return np.argmax(g, axis=0)
    
    def test_error(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = np.count_nonzero(y_pred == y_test)
        return 1 - correct/len(y_test)


    


