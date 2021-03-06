import pickle
import time
import numpy as np
from transformer.transformer_class import Transformer
from transformer.transformer_tools import create_masks
import tensorflow as tf
import io
from pprint import pprint


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# INITIALIZERS

with open('./resources/encoded_X.pickle', 'rb') as f:
    two_way_X = pickle.load(f)
with open('./resources/encoded_y.pickle', 'rb') as f:
    two_way_y = pickle.load(f)

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

'''
ORIGINALS
n, d = 2048, 512
pos_encoding = positional_encoding(n, d)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
'''

EPOCHS = 21
num_layers = 4
d_model = 256
dff = 1024
num_heads = 8
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_accuracy')

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=len(two_way_X.d) // 2 + 1,
    target_vocab_size=len(two_way_y.d) // 2 + 1,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    # pprint(vars(ckpt))
    print('Latest checkpoint restored!!')


def get_encoder_emb(two_way_X):
    weights = transformer.tokenizer.embedding.get_weights()[0]
    vocab = two_way_X

    out_v = io.open('./training_data/vectors_enc.tsv', 'w', encoding='utf-8')
    out_m = io.open('./training_data/metadata_enc.tsv', 'w', encoding='utf-8')
    for index in range(len(vocab.d) // 2):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(vocab.get(index) + "\n")
    out_v.close()
    out_m.close()

def get_decoder_emb(two_way_y):
    weights = transformer.decoder.embedding.get_weights()[0]
    vocab = two_way_y

    out_v = io.open('./training_data/vectors_dec.tsv', 'w', encoding='utf-8')
    out_m = io.open('./training_data/metadata_dec.tsv', 'w', encoding='utf-8')
    for index in range(len(vocab.d) // 2):
        if index == 0:
            continue  # skip 0, it's padding.
        print(index)
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(vocab.get(index) + "\n")
    out_v.close()
    out_m.close()

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def fit(train_batches, val_batches):
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)
            inp_val, tar_val = val_batches[batch % len(val_batches)]
            validation_step(inp_val, tar_val)

            if batch % 25 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch}'
                    f' - Loss {train_loss.result():.4f}'
                    f' - Accuracy {train_accuracy.result():.4f}'
                    f' - Val Loss {val_loss.result():.4f}'
                    f' - Val Accuracy {val_accuracy.result():.4f}'
                )

                train_accuracies.append(train_accuracy.result())
                val_accuracies.append(val_accuracy.result())
                train_losses.append(train_loss.result())
                val_losses.append(val_loss.result())

                if (epoch + 1) % 3 == 0:
                    ckpt_save_path = ckpt_manager.save()
                    print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    return np.array(train_accuracies), np.array(val_accuracies), np.array(train_losses), np.array(val_losses)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


@tf.function(input_signature=val_step_signature)
def validation_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    predictions, _ = transformer(inp, tar_inp,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)
    val_loss(loss)
    val_accuracy(accuracy_function(tar_real, predictions))
