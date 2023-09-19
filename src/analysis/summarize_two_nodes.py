import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

# Matplotlib Color map
c = ["grey", "red", "purple", "blue"]
v = [0, 0.25, 0.5, 1.0]
l = list(zip(v, c))
cmap = LinearSegmentedColormap.from_list("masks", l, N=256)


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
        dev_y.append(data["knn dev acc"].iloc[0])
        test_y.append(data["knn test acc"].iloc[0])
        sparsity.append(data["L0 Norm"].iloc[0] / data["L0 Max"].iloc[0])

    # Save the loss plot for train and test
    plt.figure()
    plt.plot(x, dev_y, label="Train set KNN Acc")
    plt.plot(x, test_y, label="Test set KNN Acc")
    if sparsity:
        plt.plot(x, sparsity, label="Sparsity")
    plt.title(figtitle)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/% Neurons")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))


def get_models_over_checkpoints(indir, operation, position):
    checkpoints = os.listdir(indir)

    models = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(indir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        data = data[data["probe index"] == position]
        models.append(data["model_id"].iloc[0])
    return models, x


def compute_model_overlap(a_model_dir, b_model_dir, a_models, b_models, epochs, outdir):
    os.makedirs(os.path.join(outdir, "Heatmaps"), exist_ok=True)
    iou = defaultdict(list)
    iom = defaultdict(list)

    for i in range(len(a_models)):
        epoch = epochs[i]

        a_path = os.path.join(a_model_dir, str(epoch), a_models[i] + ".pt")
        b_path = os.path.join(b_model_dir, str(epoch), b_models[i] + ".pt")

        a_mod = torch.load(a_path)
        b_mod = torch.load(b_path)

        for k in a_mod.keys():
            if "weight_mask_params" in k and "to_hook" in k:
                intersection = torch.logical_and(a_mod[k] > 0, b_mod[k] > 0)
                sum_intersection = torch.sum(intersection)
                union = torch.sum(torch.logical_or(a_mod[k] > 0, b_mod[k] > 0))
                minimum = np.min(
                    [torch.sum(a_mod[k] > 0).cpu(), torch.sum(b_mod[k] > 0).cpu()]
                )

                iou[k].append((sum_intersection / union).cpu())
                iom[k].append((sum_intersection / minimum).cpu())

                # Make Heatmap
                hm = torch.full(a_mod[k].shape, 0.0)
                hm[a_mod[k] > 0] = 0.25
                hm[b_mod[k] > 0] = 1.0
                hm[intersection] = 0.5

                hm = hm.reshape(16, -1)
                plt.figure()
                plt.imshow(hm, cmap=cmap, vmin=0, vmax=1.0)
                plt.title(f"Mask Overlap: {k} - {epoch}")

                outpath = os.path.join(outdir, "Heatmaps", f"{k}-{epoch}.png")
                plt.savefig(outpath)

    for k in iou.keys():
        # Line graph of IoU and IoM
        plt.figure()
        plt.plot(epochs, iou[k], label="IoU")
        plt.plot(epochs, iom[k], label="IoM")
        plt.title(k)
        plt.xlabel("Epochs")
        plt.ylabel("Percent")
        plt.legend()

        plt.savefig(os.path.join(outdir, f"{k}_overlap.png"))


# Directories
a_results_directory = "../../Results/Probes/Two_Nodes/amod29_bmod31/A/Mask"
b_results_directory = "../../Results/Probes/Two_Nodes/amod29_bmod31/B/Mask"

a_results_directory_full = "../../Results/Probes/Two_Nodes/amod29_bmod31/A/Full"
b_results_directory_full = "../../Results/Probes/Two_Nodes/amod29_bmod31/B/Full"

a_model_directory = "../../Model/Probes/Two_Nodes/amod29_bmod31/A/Mask"
b_model_directory = "../../Model/Probes/Two_Nodes/amod29_bmod31/B/Mask"

outdir = "./Two_Nodes/"
create_knn_chart(
    a_results_directory,
    "A Node",
    outdir,
    outfile="A_Node_KNN.png",
    operation="attn",
    position=2,
    sparsity=True,
)
create_knn_chart(
    b_results_directory,
    "B Node",
    outdir,
    outfile="B_Node_KNN.png",
    operation="attn",
    position=2,
    sparsity=True,
)

create_knn_chart(
    a_results_directory_full,
    "Full A Node",
    outdir,
    outfile="A_Node_Full_KNN.png",
    operation="attn",
    position=0,
    sparsity=True,
)
create_knn_chart(
    b_results_directory_full,
    "Full B Node",
    outdir,
    outfile="B_Node_Full_KNN.png",
    operation="attn",
    position=1,
    sparsity=True,
)

a_models, _ = get_models_over_checkpoints(a_results_directory, "attn", 2)
b_models, epochs = get_models_over_checkpoints(b_results_directory, "attn", 2)

compute_model_overlap(
    a_model_directory, b_model_directory, a_models, b_models, epochs, outdir
)
