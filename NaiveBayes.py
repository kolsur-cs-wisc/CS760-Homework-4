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

        self.y_labels = set(self.y)
        self.class_count = len(self.y_labels)
        self.prior, self.log_prior = {}, {}
        self.theta, self.log_theta= {}, {}

        for label in self.y_labels:
            X_label = self.X[self.y == label]
            self.prior[label] = (len(X_label) + self.alpha) / (m + self.class_count*self.alpha)
            self.log_prior[label] = np.log(self.prior[label])
            self.theta[label] = np.zeros(n)
            total_chars = 0
            for countMap in X_label:
                for char, count in countMap.items():
                    self.theta[label][self.vocab.index(char)] += count
                total_chars += sum(countMap.values())
            self.theta[label] = (self.theta[label] + self.alpha) / (total_chars + n*self.alpha)
            print(self.theta[label], sum(self.theta[label]))
            self.log_theta[label] = np.log(self.theta[label])

    def predict(self, X_test):
        return [self.predict_single(word_map) for word_map in X_test]

    def predict_single(self, X_word_map):
        likelihood_prob = {}
        posterior = {}

        for label in self.y_labels:
            label_log_likelihood = self.log_theta[label]
            curr_likelihood = 0
            for char, count in X_word_map.items():
                curr_likelihood += count * label_log_likelihood[self.vocab.index(char)]
            likelihood_prob[label] = np.exp(curr_likelihood)
            posterior[label] = curr_likelihood + self.log_prior[label]

        print(likelihood_prob)
        return max(posterior, key=posterior.get)
    
    def accuracy(self, X_test, y_test):
        total_test = len(y_test)
        y_pred = self.predict(X_test)
        error = 0
        for i in range(total_test):
            if y_pred[i] != y_test[i]:
                error += 1

        return 1 - (error/total_test)