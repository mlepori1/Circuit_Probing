import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap


def create_knn_chart(indir, figtitle, outdir, outfile, sparsity=False):
    os.makedirs(outdir, exist_ok=True)

    data = pd.read_csv(os.path.join(indir, "results.csv"))
    dev_y = data["knn dev acc"]
    test_y = data["knn test acc"]
    sparsity_data = data["L0 Norm"] / data["L0 Max"]

    # Save the loss plot for train and test
    plt.figure()
    plt.plot(list(range(len(dev_y))), dev_y, label="Train set KNN Acc")
    plt.plot(list(range(len(dev_y))), test_y, label="Test set KNN Acc")
    if sparsity:
        plt.plot(list(range(len(dev_y))), sparsity_data, label="Sparsity")
    plt.title(figtitle)
    plt.xlabel("Model Components")
    plt.ylabel("Accuracy/% Neurons")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))


def create_ablation_chart(indir, figtitle, outdir, outfile):
    os.makedirs(outdir, exist_ok=True)

    data = pd.read_csv(os.path.join(indir, "results.csv"))

    iid_baseline = data[" vanilla acc IID"][0]
    gen_baseline = data[" vanilla acc Gen"][0]

    ablated_iid = data[" ablated acc IID"]
    ablated_gen = data[" ablated acc Gen"]

    random_iid = data["random ablate acc mean IID"]
    random_gen = data["random ablate acc mean Gen"]

    iid_std = data["random ablate acc std IID"]
    gen_std = data["random ablate acc std Gen"]

    x = np.array(list(range(0, int(len(ablated_iid) * 3), 3)))

    # Save the loss plot for train and test
    plt.figure(figsize=(50, 10))
    plt.bar(x, ablated_iid, width=0.5, label="Ablate IID")
    plt.bar(x + 0.5, random_iid, yerr=iid_std, width=0.5, label="Random Abl IID")

    plt.bar(x + 1, ablated_gen, width=0.5, label="Ablate Gen")
    plt.bar(x + 1.5, random_gen, yerr=gen_std, width=0.5, label="Random Abl Gen")

    plt.axhline(iid_baseline)
    plt.axhline(gen_baseline)
    plt.xticks(x + 0.75, labels=list(range(len(x))))
    plt.ylim(0.4, 0.9)
    plt.title(figtitle)
    plt.xlabel("Model Components")
    plt.ylabel("Agreement Accuracy After Ablation")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))


# Directories
results_directory = "../../Results/Probes/SV_Agr/small"

outdir = "./SV_Agr/small"
create_knn_chart(
    results_directory,
    "Agreement KNN Accuracy",
    outdir,
    outfile="Agreement_KNN.png",
    sparsity=True,
)
create_ablation_chart(
    results_directory,
    "Agreement Ablation Accuracy",
    outdir,
    outfile="Agreement_Ablation.png",
)

# Directories
results_directory = "../../Results/Probes/SV_Agr/medium"

outdir = "./SV_Agr/medium"
create_knn_chart(
    results_directory,
    "Agreement KNN Accuracy",
    outdir,
    outfile="Agreement_KNN.png",
    sparsity=True,
)
create_ablation_chart(
    results_directory,
    "Agreement Ablation Accuracy",
    outdir,
    outfile="Agreement_Ablation.png",
)
