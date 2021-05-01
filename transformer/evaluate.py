import copy
import re
import tensorflow as tf
import numpy as np
from transformer.transformer_tools import create_masks
from transformer.transformer_training import transformer
from main import detokenize, tokenize, make_human_understandable
from Levenshtein import distance as levenshtein_distance


def check_next_syl(two_way_y, syl, output, sentence):
    syl = detokenize(two_way_y, syl)
    output = detokenize(two_way_y, output)
    output = re.sub(r'<syl>', '', output)
    sentence = re.sub(output, '', sentence)
    if syl == '<syl>' or syl == '<end>' or syl == '<c>' or re.search(r'^' + syl, sentence):
        return True
    return False


def evaluate(sentence, two_way_X, two_way_y, max_length=40):
    encoder_input = tf.cast(tf.convert_to_tensor([tokenize(two_way_X, sentence)]), tf.int64)
    start, end = two_way_y.get('<start>'), two_way_y.get('<end>')
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)
    output = tf.cast(output, tf.int64)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        """ CHANGE START: decomment only one of them """
        """ 1 - ORIGINAL"""
        predicted_id = tf.argmax(predictions, axis=-1)
        """ 2 - MODIFIED"""
        """
        predicted_id_orig = tf.argmax(predictions, axis=-1)
        count = 0
        while True:
            if count == 5:
                predicted_id = predicted_id_orig
                break
            predicted_id = tf.argmax(predictions, axis=-1)
            # concatentate the predicted_id to the output which is given to the decoder as its input.
            if check_next_syl(two_way_y, copy.deepcopy(predicted_id), output, sentence):
                break
            predictions = predictions.numpy()
            predictions[:, :, predicted_id.numpy()] = -100
            predictions = tf.convert_to_tensor(predictions)
            count += 1
            """
        """ CHANGE STOP """

        output = tf.concat([output, predicted_id], axis=-1)
        # return the result if the predicted_id is equal to the end token
        if predicted_id == end:
            break
    # output.shape (1, tokens)
    text = detokenize(two_way_y, output)
    return text, attention_weights


def evaluate_test(X_test, y_test, two_way_X, two_way_y):
    print(len(X_test))
    distances = []
    for query_sent, true_sent in zip(X_test[0:30], y_test[0:30]):
        pred_text, attention_w = evaluate(query_sent, two_way_X, two_way_y)
        pred_text = make_human_understandable(pred_text)
        true_sent = make_human_understandable(true_sent)
        print(f"pred: {pred_text}\norig: {true_sent}")
        lev = levenshtein_distance(pred_text, true_sent)
        lower = abs(len(pred_text) - len(true_sent))
        upper = max(len(pred_text), len(true_sent))
        distances.append((lev - lower) / (upper - lower))
    print(1 - np.mean(distances))
