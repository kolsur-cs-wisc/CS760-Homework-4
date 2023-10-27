import numpy as np
from NaiveBayes import Naive_Bayes

def get_vocab():
    return ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ')

def bag_of_words(document):
    vocab = get_vocab()
    word_count = dict.fromkeys(vocab, 0)
    
    for sentence in document:
        for word in sentence:
            if word in vocab:
                word_count[word] = word_count.get(word) + 1

    return word_count


def main():
    default_path = "languageID/"
    languages = ['e', 'j', 's']
    X_train = []
    y_train = []
    
    for lang in languages:
        for i in range(10):
            document = np.loadtxt(default_path+f'{lang}{i}.txt', delimiter='\t', dtype=str)
            words = bag_of_words(document)
            X_train.append(words)
            y_train.append(lang)

    nb_model = Naive_Bayes(np.array(X_train), np.array(y_train), get_vocab())
    nb_model.train()

    test_document = np.loadtxt(default_path+'e10.txt', delimiter='\t', dtype=str)
    test_words = bag_of_words(test_document)
    print(f'Bag of Words e10: {test_words}')

    print(nb_model.predict([test_words]))

    X_test = []
    y_test = []

    for lang in languages:
        for i in range(10, 20):
            document = np.loadtxt(default_path+f'{lang}{i}.txt', delimiter='\t', dtype=str)
            words = bag_of_words(document)
            X_test.append(words)
            y_test.append(lang)
    
    print(nb_model.accuracy(X_test, y_test))

if __name__ == '__main__':
    main()