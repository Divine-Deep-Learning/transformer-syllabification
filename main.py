import re
import pickle
import sys

import numpy as np
from two_way import TwoWay

from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformer import transformer_training, evaluate
import text_precleaner as text_precleaner_

sys.path.insert(1, './syllabificator')


def encode_dataset(X, y):
    """
        Creates integer encoded version of the dataset.
    """
    encoded_X = TwoWay()
    encoded_set = []
    for row in X:
        tmp_row = re.sub(r'<start>|<end>|<syl>', r'', row)
        [[encoded_set.append(c) for c in w] for w in tmp_row.split('<s>')]
    encoded_set += ['<start>', '<end>', '<s>']
    encoded_set = set(encoded_set)
    [encoded_X.add(i + 1, w) for i, w in enumerate(encoded_set)]
    with open('resources/encoded_X.pickle', 'wb') as handle:
        pickle.dump(encoded_X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    encoded_y = TwoWay()
    encoded_set = []
    for row in y:
        tmp_row = re.sub(r'<syl>', r'<s>', row)
        tmp_row = re.sub(r'<start>|<end>|<c>', r'', tmp_row)
        [[encoded_set.append(c) for c in w] for w in tmp_row.split('<s>')]
    encoded_set += ['<syl>', '<s>', '<start>', '<end>', '<c>']
    encoded_set = set(encoded_set)
    try:
        encoded_set.remove("")
    except KeyError:
        pass
    [encoded_y.add(i + 1, w) for i, w in enumerate(encoded_set)]
    with open('resources/encoded_y.pickle', 'wb') as handle:
        pickle.dump(encoded_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data saved successfully!")


def tokenize(two_way, line, X=True):
    spaced_line = re.sub(r'<', r' <', line)
    spaced_line = re.sub(r'>', r'> ', spaced_line)
    spaced_line = re.sub(r'^ | $', r'', spaced_line)
    spaced_line = re.sub(r'[ ]+', r' ', spaced_line)
    spaced_line = spaced_line.split(' ')
    while True:
        try:
            spaced_line.remove('')
        except ValueError:
            break
    if X:
        tok_X = []
        for w in spaced_line:
            if w in ['<start>', '<end>', '<s>', '<syl>', '<c>']:
                tok_X.append(two_way.get(w))
            else:
                [tok_X.append(two_way.get(c)) for c in w]
        return tok_X
    else:
        return [two_way.get(e) for e in spaced_line]


def detokenize(two_way, line):
    sentence = [two_way.get(e.numpy()) for e in line[0]]
    return ''.join(sentence)


def detokenize_(two_way, line):
    sentence = [two_way.get(e) for e in line]
    return ''.join(sentence)


def make_human_understandable(sentence):
    sentence = re.sub(r'<start>|<end>', r'', sentence)
    sentence = re.sub(r'<syl>', r'|', sentence)
    sentence = re.sub(r'<s>', r' ', sentence)
    return sentence


def tokenize_pairs(X, y):
    X_tok = [tokenize(two_way_X, l, X=True) for l in X]
    y_tok = [tokenize(two_way_y, l, X=True) for l in y]
    return X_tok, y_tok


def make_batches(X_y_tok, batch_size):
    batches = []
    X_tok, y_tok = X_y_tok
    for i in range(0, len(X_tok), batch_size):
        if batch_size + i < len(X_tok):
            batches.append((tf.cast(tf.ragged.constant(X_tok[i:i + batch_size]), tf.int64).to_tensor(),
                            (tf.cast(tf.ragged.constant(y_tok[i:i + batch_size]), tf.int64).to_tensor())))
    return batches


if __name__ == '__main__':
    X = np.loadtxt("resources/X_cesura.csv", dtype=str, delimiter=',', encoding='utf-8')
    y = np.loadtxt("resources/y_cesura.csv", dtype=str, delimiter=',')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.03, random_state=0)
    # encode_dataset(X, y)

    with open('resources/encoded_X.pickle', 'rb') as f:
        two_way_X = pickle.load(f)
    with open('resources/encoded_y.pickle', 'rb') as f:
        two_way_y = pickle.load(f)

    BATCH_SIZE = 64
    train_batches = make_batches(tokenize_pairs(X_train, y_train), batch_size=BATCH_SIZE)
    val_batches = make_batches(tokenize_pairs(X_val, y_val), batch_size=BATCH_SIZE)

    # train_accuracies, val_accuracies, train_losses, val_losses = transformer_training.fit(train_batches, val_batches)
    # np.save('./training_data/train_accuracies.npy', train_accuracies)
    # np.save('./training_data/val_accuracies.npy', val_accuracies)
    # np.save('./training_data/train_losses.npy', train_losses)
    # np.save('./training_data/val_losses.npy', val_losses)

    # transformer_training.get_encoder_emb(two_way_X)
    # transformer_training.get_decoder_emb(two_way_y)

    # evaluate.evaluate_test(X_val, y_val, two_way_X, two_way_y)

    # CUSTOM QUERY

    X_ar = ["Le donne, i cavallier, l'arme, gli amori,",
            "le cortesie, l'audaci imprese io canto"]

    for query_sent in X_ar:
        q_cleaned = text_precleaner_.sub_cleaner(query_sent)
        pred_text, _ = evaluate.evaluate(q_cleaned, two_way_X, two_way_y)
        pred_text = make_human_understandable(pred_text)
        print(pred_text)
