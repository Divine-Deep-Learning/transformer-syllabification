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


EPOCHS = 21


def print_acc_loss():
    train_accuracies = np.load('./training_data/train_accuracies.npy')
    val_accuracies = np.load('./training_data/val_accuracies.npy')
    train_losses = np.load('./training_data/train_losses.npy')
    val_losses = np.load('./training_data/val_losses.npy')

    x = np.linspace(1, EPOCHS, len(train_accuracies))

    plt.title("Accuracies")
    plt.plot(x, train_accuracies)
    plt.plot(x, val_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "val_acc"])
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./training_data/Accuracy.png')
    plt.show()
    print()
    plt.title("Losses")
    plt.plot(x, train_losses)
    plt.plot(x, val_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"])
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./training_data/Loss.png')
    plt.show()

print_acc_loss()


