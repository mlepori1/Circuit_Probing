import os

import matplotlib.pyplot as plt
import pandas as pd


def create_probe_chart(
    indir, figtitle, outdir, outfile, operation, position
):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(indir)

    dev_y = []
    test_y = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(indir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        data = data[data["probe index"] == position]
        dev_y.append(data["train acc"][0])
        test_y.append(data["test acc"][0])

    # Save the loss plot for train and test
    plt.figure()
    plt.plot(x, dev_y, label="Train set Probe Acc")
    plt.plot(x, test_y, label="Test set Probe Acc")
    plt.ylim(0, 1.05)
    plt.title(figtitle)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))

def create_knn_chart(
    indir, figtitle, outdir, outfile, operation, position, sparsity=False
):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(indir)

    dev_y = []
    test_y = []
    sparsity = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(indir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        data = data[data["probe index"] == position]
        dev_y.append(data["knn dev acc"][0])
        test_y.append(data["knn test acc"][0])
        sparsity.append(data["L0 Norm"][0] / data["L0 Max"][0])

    # Save the loss plot for train and test
    plt.figure()
    plt.plot(x, dev_y, label="Train set KNN Acc")
    plt.plot(x, test_y, label="Test set KNN Acc")
    if sparsity:
        plt.plot(x, sparsity, label="Sparsity")
    plt.title(figtitle)
    plt.ylim(0, 1.05)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/% Neurons")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))


indir = "../../Results/Probes/Grokking/a2_b/Mask/a2"
figtitle = "Attn Layer 0 - A2"
outdir = "./Grokking/a2_b/"
outfile = "a2_knn.png"
create_knn_chart(indir, figtitle, outdir, outfile, "attn", 2, sparsity=True)

indir = "../../Results/Probes/Grokking/a2_b/Mask/b2"
figtitle = "Attn Layer 0 - B2"
outdir = "./Grokking/a2_b/"
outfile = "b2_knn.png"
create_knn_chart(indir, figtitle, outdir, outfile, "attn", 2, sparsity=True)

indir = "../../Results/Probes/Grokking/a2_b/Probe/a2"
figtitle = "Probing: Attn Layer 0 - A2"
outdir = "./Grokking/a2_b/"
outfile = "a2_probe.png"
create_probe_chart(indir, figtitle, outdir, outfile, "attn", 2)

indir = "../../Results/Probes/Grokking/a2_b/Probe/b2"
figtitle = "Probing: Attn Layer 0 - B2"
outdir = "./Grokking/a2_b/"
outfile = "b2_probe.png"
create_probe_chart(indir, figtitle, outdir, outfile, "attn", 2)