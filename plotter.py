import numpy as np
import matplotlib.pyplot as plt


def plot_attention_head(in_tokens, translated_tokens, attention):
    translated_tokens = translated_tokens[1:]
    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))
    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)
    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


def print_acc_loss():
    train_accuracies = np.load('./training_data/train_accuracies.npy')
    val_accuracies = np.load('./training_data/val_accuracies.npy')
    train_losses = np.load('./training_data/train_losses.npy')
    val_losses = np.load('./training_data/val_losses.npy')

    plt.title("Accuracies")
    plt.plot(range(0, 25 * len(train_accuracies), 25), train_accuracies)
    plt.plot(range(0, 25 * len(train_accuracies), 25), val_accuracies)
    plt.xlabel("Num batch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "val_acc"])
    plt.show()

    plt.title("Loss")
    plt.plot(range(0, 25 * len(train_accuracies), 25), train_losses)
    plt.plot(range(0, 25 * len(train_accuracies), 25), val_losses)
    plt.xlabel("Num batch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"])
    plt.show()


# print_acc_loss()


