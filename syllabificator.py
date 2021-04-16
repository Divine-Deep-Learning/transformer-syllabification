import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# import tensorflow_datasets as tfds
# import tensorflow_text as text
import tensorflow as tf
from tensorflow import keras

MASK_VALUE = -1
MAX_WORD_LEN = None
GEN_INDEX = 0
ENCODER = {c: i for i, c in enumerate("abcdefghilmnopqrstuvzàèéìòù'")}
DECODER = {i: c for i, c in enumerate("abcdefghilmnopqrstuvzàèéìòù'")}


def encode_X(array):
    encoded_set = []
    for row in array:
        tmp_row = re.sub(r'<start>|<end>|<syl>', r'', row)
        [encoded_set.append(w) for w in tmp_row.split('<s>')]
    return list(set(encoded_set))


def decode(word):
    return [DECODER[i] if i != -1 else '' for i in word]


def one_hot(word, dataset):
    clean_w = word.split('-')  # for syllabs
    index_array = []
    for syl in clean_w:
        i = 0
        for e in dataset:
            if syl == e:
                index_array.append(i)
            i += 1

    word_one_hot_enc = np.zeros(len(dataset))
    for index in index_array:
        word_one_hot_enc[index] = 1
    return word_one_hot_enc


def create_syllabs_set(y):
    all_syllabs = []
    for word in y:
        for syl in word.split('-'):
            all_syllabs.append(syl)
    return dict(set(all_syllabs))


def padder_encoder(dataset, one_hot=True):
    """
    :param dataset:
    :return: padded version of the dataset
    """
    global MAX_WORD_LEN
    MAX_WORD_LEN = max([len(w) for w in dataset])
    print(MAX_WORD_LEN)
    encoded_train = [encode(w) for w in dataset]

    if one_hot:
        final_data = []

        for el in encoded_train:
            padded_word = np.zeros(shape=(len(ENCODER), MAX_WORD_LEN))
            for i, e in enumerate(el):
                padded_word[e, i] = 1
            padded_word[:, len(el):MAX_WORD_LEN] = MASK_VALUE
            final_data.append(padded_word.T)
        return np.array(final_data)
    else:
        return keras.preprocessing.sequence.pad_sequences(encoded_train,
                                                          maxlen=MAX_WORD_LEN,
                                                          value=MASK_VALUE,
                                                          padding='post')


def generator(batchsize, X, y, syllabs_set):
    """
    :param batchsize:
    :param X: has to be padded
    :param y:
    :return:
    """
    global GEN_INDEX
    while True:
        index = GEN_INDEX % len(X)
        index_sup = index + batchsize
        inputs = X[index:index_sup]
        if index_sup >= len(X):
            index_sup = index + batchsize - len(X)
            inputs = np.concatenate((X[index:], X[:index_sup]))
        labels_ = []
        if index_sup != index + batchsize:
            for i in range(index, len(X)):
                labels_.append(one_hot(y[i], syllabs_set))
            for i in range(0, index_sup):
                labels_.append(one_hot(y[i], syllabs_set))
        else:
            for i in range(index, index_sup):
                labels_.append(one_hot(y[i], syllabs_set))
        GEN_INDEX += batchsize
        yield np.array(inputs), np.array(labels_)


if __name__ == '__main__':
    X_train = np.loadtxt("resources/X_train.csv", dtype=str,
                         delimiter=',', encoding='utf-8')
    y_train = np.loadtxt("resources/y_train.csv", dtype=str, delimiter=',')

    print(encode_X(X_train))
    print(len(encode_X(X_train)))
