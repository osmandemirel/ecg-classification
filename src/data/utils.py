"""
Augmentation methods to be applied can be stored here
Other data related utilities will be added when needed
"""
import matplotlib.pyplot as plt

def read_metrics(path):
    with open(path, "r") as f:
        t = f.read()
    return eval(t)


def process_validation(metrics):
    val_losses, val_accs = [], []
    val_interval = int(len(metrics["train_loss"]) / len(metrics["val_loss"]))
    for i in range(len(metrics["val_loss"])):
        val_losses += [metrics["val_loss"][i]] * int(val_interval)
        val_accs += [metrics["val_acc"][i]] * int(val_interval)
    return val_losses, val_accs


def plot_metric(train_metric, val_metric, metric="loss"):
    plt.plot(train_metric, label=f"train_{metric}")
    plt.plot(val_metric, label=f"validation_{metric}")
    plt.title(f'{metric} curve')
    plt.show()
 
# usage for colab notebook
path = "/content/drive/MyDrive/results/metrics.txt"
metrics = read_metrics(path)
val_loss, val_acc = process_validation(metrics)

plot_metric(metrics["train_loss"], val_loss, metric="loss")
plot_metric(metrics["train_acc"], val_acc, metric="accuracy")
