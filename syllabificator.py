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
    encoded_list = [[i, el] for i, el in enumerate(encoded_set)]
    with open('resources/encoded_X.pickle', 'wb') as handle:
        pickle.dump(encoded_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    encoded_set = []
    for row in y:
        tmp_row = re.sub(r'<syl>', r'<s>', row)
        [encoded_set.append(w) for w in tmp_row.split('<s>')]
    encoded_set += ['<syl>', '<s>']
    encoded_set = set(encoded_set)
    encoded_set.remove("")
    encoded_list = [[i, el] for i, el in enumerate(encoded_set)]
    with open('resources/encoded_y.pickle', 'wb') as handle:
        pickle.dump(encoded_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    # encode_dataset(X, y)
    with open('resources/encoded_X.pickle', 'rb') as f:
        encode_X = pickle.load(f)
    with open('resources/encoded_y.pickle', 'rb') as f:
        encode_y = pickle.load(f)

    print(encode_X[:5])
