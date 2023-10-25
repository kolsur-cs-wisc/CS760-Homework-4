import numpy as np
import pandas as pd

class ThreeLayerNN():
    def __init__(self, X_train, y_train, lr = 0.05, epochs = 5000):
        self.X = X_train
        self.y = y_train
        self.lr = lr
        self.epochs = epochs
        self.m, self.n_X = X_train.shape
        self.n_y = y_train.shape[0]
        self.n_h = 300
        self.initialize_weights()

    def initialize_weights(self):
        self.W1 = np.random.uniform(-1, 1, (self.n_h, self.n_X))
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.uniform(-1, 1, (self.n_y, self.n_h))
        self.b2 = np.zeros((self.n_y, 1))

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def delta_sigmoid(self, value):
        return self.sigmoid(value) * (1 - self.sigmoid(value))

    def train(self):
        for i in range(self.epochs):
            Z1 = np.matmul(self.W1, self.X.T) + self.b1                   # (64,784)*(784,42000) + (64,1) ---> (64,42000)
            A1 = self.sigmoid(Z1)                                       # (64,42000)

            Z2 = np.matmul(self.W2, A1) + self.b2                               # (10,64) * (64,42000) + (10,1) ---> (10,42000)
            A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)           # (10,42000)

            cost = self.loss_func(A2, self.y)                      # (10,42000)

            dZ2 = A2 - self.y                                        # (10, 42000)
            dW2 = (1/self.m) * np.matmul(dZ2, A1.T)                    # (10,64)
            db2 = (1/self.m) * np.sum(dZ2, axis=1, keepdims=True)      # (10, 1)

            dA1 = np.matmul(self.W2.T, dZ2)                             # (64, 42000)
            dZ1 = dA1 * self.sigmoid(Z1) * (1 - self.sigmoid(Z1))            # (64, 42000)
            dW1 = (1./self.m) * np.matmul(dZ1, self.X)                # (64,784)  
            db1 = (1./self.m) * np.sum(dZ1, axis=1, keepdims=True)      # (64,1)

            # Now to update the matrices of W1, W2, b1 and b2.
            self.W2 = self.W2 - self.lr * dW2
            self.b2 = self.b2 - self.lr * db2
            self.W1 = self.W1 - self.lr * dW1
            self.b1 = self.b1 - self.lr * db1
        print("Final cost:", cost) 

    def loss_func(self, y_pred, y_label):
        return -np.sum(np.multiply(y_label, np.log(y_pred)))
    
    def predict(self, X_test):
        Z1 = np.matmul(self.W1, X_test.T) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

        predictions = np.argmax(A2, axis=0)
        return predictions
    
    def test_error(self, X_test, y_test):
        y_pred = self.predict(X_test)
        correct = np.count_nonzero(y_pred == y_test)
        return 1 - correct/len(y_test)


    


