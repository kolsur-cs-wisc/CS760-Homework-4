import numpy as np
import pandas as pd

class Naive_Bayes():
    def __init__(self, X_train, y_train, vocab, alpha=0.5):
        self.X = X_train
        self.y = y_train
        self.vocab = vocab
        self.alpha = alpha
    
    def train(self):
        m, n = self.X.size, len(self.X[0])

        y_types = set(self.y)
        total = len(self.y)
        prior = np.zeros(len(y_types))

        for i, label in enumerate(y_types):
            prior[i] = (len(self.X[self.y == label])+self.alpha)/(total+len(y_types)*self.alpha)
        print(prior)