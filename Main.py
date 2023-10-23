import numpy as np
import pandas as pd

def bag_of_words(document):
    vocab = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ')
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
            document = np.loadtxt(default_path+f'{lang}{i}.txt', delimiter='\n', dtype=str)
            words = bag_of_words(document)
            X_train.append(words)
            y_train.append(lang)

if __name__ == '__main__':
    main()