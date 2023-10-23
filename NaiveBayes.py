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
        class_count = len(y_types)
        prior, log_prior = np.zeros(class_count), np.zeros(class_count)

        likelihood, log_likelihood= {}, {}

        for i, label in enumerate(y_types):
            X_label = self.X[self.y == label]
            prior[i] = (len(X_label) + self.alpha) / (total + class_count*self.alpha)
            likelihood[label] = np.zeros(n)
            total_chars = 0
            for countMap in X_label:
                for char, count in countMap.items():
                    likelihood[label][self.vocab.index(char)] += count
                total_chars += sum(countMap.values())
            likelihood[label] = (likelihood[label] + self.alpha) / (total_chars + n*self.alpha)
            log_likelihood[label] = np.log(likelihood[label])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        log_prior =  np.log(prior)