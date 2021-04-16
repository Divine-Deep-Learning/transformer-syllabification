import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

# import tensorflow_datasets as tfds
# import tensorflow_text as text
import tensorflow as tf

MASK_VALUE = -1
MAX_WORD_LEN = None
GEN_INDEX = 0


def encode_dataset(X, y):
    """
        Creates integer encoded version of the dataset.
    """
    encoded_set = []
    for row in X:
        tmp_row = re.sub(r'<start>|<end>|<syl>', r'', row)
        [encoded_set.append(w) for w in tmp_row.split('<s>')]
    encoded_set += ['<start>', '<end>', '<s>']
    encoded_set = set(encoded_set)
    encoded_array = np.array([[i, el] for i, el in enumerate(encoded_set)], dtype='object')
    np.save("resources/encoded_X.npy", encoded_array, allow_pickle=True)

    encoded_set = []
    for row in y:
        tmp_row = re.sub(r'<syl>', r'<s>', row)
        [encoded_set.append(w) for w in tmp_row.split('<s>')]
    encoded_set += ['<syl>', '<s>']
    encoded_set = set(encoded_set)
    encoded_set.remove("")
    encoded_array = np.array([[i, el] for i, el in enumerate(encoded_set)], dtype='object')
    np.save("resources/encoded_y.npy", encoded_array, allow_pickle=True)

    print("Data saved successfully!")


def decode(word, dict):
    for key, value in dict.items():
        if word == value:
            return key
    return "key doesn't exist"


def tokenize(dict, line):
    spaced_line = re.sub(r'<', r' <', line)
    spaced_line = re.sub(r'>', r'> ', spaced_line)
    spaced_line = re.sub(r'^ | $', r'', spaced_line)
    spaced_line = line.split(' ')


if __name__ == '__main__':
    X = np.loadtxt("resources/X.csv", dtype=str, delimiter=',', encoding='utf-8')
    y = np.loadtxt("resources/y.csv", dtype=str, delimiter=',', encoding='utf-8')

    encode_dataset(X, y)

    encode_X = np.load("resources/encoded_X.npy", allow_pickle=True)
    encode_y = np.load("resources/encoded_y.npy", allow_pickle=True)

    print(encode_X[:5], len(encode_X))
    print(encode_y[:5], len(encode_y))
