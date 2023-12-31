import numpy as np

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
            self.log_prior[label] = np.log10(self.prior[label])
            print(f'Prior {label} = {self.prior[label]}, Log10 Space = {self.log_prior[label]}')
            self.theta[label] = np.zeros(n)
            total_chars = 0
            for countMap in X_label:
                for char, count in countMap.items():
                    self.theta[label][self.vocab.index(char)] += count
                total_chars += sum(countMap.values())
            self.theta[label] = (self.theta[label] + self.alpha) / (total_chars + n*self.alpha)
            print(f'Multinomial Parameter Theta {label}: ', self.theta[label])
            self.log_theta[label] = np.log10(self.theta[label])

    def predict(self, X_test):
        return [self.predict_single(word_map) for word_map in X_test]

    def predict_single(self, X_word_map):
        likelihood_prob = {}
        posterior = {}

        for label in self.y_labels:
            label_log_likelihood = self.log_theta[label]
            curr_likelihood = 0
            total_char = 0
            for char, count in X_word_map.items():
                curr_likelihood += count * label_log_likelihood[self.vocab.index(char)]
                total_char += count
            likelihood_prob[label] = curr_likelihood
            posterior[label] = curr_likelihood + self.log_prior[label]

        print(f'Likelihood p(x|y) =  {likelihood_prob}')
        print(f'Posterior p(y|x) = {posterior}')
        print((np.power(10, list(likelihood_prob.values()), dtype=np.longdouble) / np.sum(np.power(10, list(posterior.values()), dtype=np.longdouble)))/3)
        return max(posterior, key=posterior.get)
    
    def accuracy(self, X_test, y_test):
        total_test = len(y_test)
        y_pred = self.predict(X_test)
        error = 0
        for i in range(total_test):
            if y_pred[i] != y_test[i]:
                error += 1

        return 1 - (error/total_test)