import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import ..utils.create_circuit_probe as create_circuit_probe

from matplotlib.colors import LinearSegmentedColormap

# Matplotlib Color map
c = ["grey", "red", "purple", "blue"]
v = [0, 0.25, 0.5, 1.0]
l = list(zip(v, c))
cmap = LinearSegmentedColormap.from_list("masks", l, N=256)


def create_knn_chart(
    indir, figtitle, outdir, outfile, operation, position, layer=0, sparsity=False
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
        data = data[data["target_layer"] == layer]
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
    plt.ylabel("Accuracy/% Weights")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))


def create_ablation_chart(
    indir, title_prefix, outdir, outfile, operation, position, component_prefix, layer=0
):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(indir)

    same = []
    different = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(indir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        data = data[data["probe index"] == position]
        data = data[data["target_layer"] == layer]
        same.append(data[f"{component_prefix} ablated acc Same"].iloc[0])
        different.append(data[f"{component_prefix} ablated acc Different"].iloc[0])

    # Save the loss plot for train and test
    plt.figure()
    plt.plot(x, same, label="LM Ablated Acc: Same Task")
    plt.plot(x, different, label="LM Ablated Acc: Different Task")
    plt.ylim(0, 1.0)
    plt.title(
        f"{title_prefix}: {operation} {component_prefix}-Position {position} LM accuracy"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(os.path.join(outdir, outfile))


def get_models_over_checkpoints(indir, operation, position, layer=0):
    checkpoints = os.listdir(indir)

    models = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(indir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        data = data[data["probe index"] == position]
        data = data[data["target_layer"] == layer]
        models.append(data["model_id"].iloc[0])
    return models, x


def compute_model_overlap(a_model_dir, b_model_dir, a_models, b_models, epochs, outdir):
    os.makedirs(os.path.join(outdir, "Heatmaps"), exist_ok=True)

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
t0_free_results_dir = "../../Results/Probes/Shared_Nodes/Task_1/Unshared"
t0_shared_results_dir = "../../Results/Probes/Shared_Nodes/Task_1/Shared"

t1_free_results_dir = "../../Results/Probes/Shared_Nodes/Task_2/Unshared"
t1_shared_results_dir = "../../Results/Probes/Shared_Nodes/Task_2/Shared"

t0_free_model_directory = "../../Model/Probes/Shared_Nodes/Task_1/Unshared"
t0_shared_model_directory = "../../Model/Probes/Shared_Nodes/Task_1/Shared"

t1_free_model_directory = "../../Model/Probes/Shared_Nodes/Task_2/Unshared"
t1_shared_model_directory = "../../Model/Probes/Shared_Nodes/Task_2/Shared"

outdir = "./Shared_Node/"
create_knn_chart(
    t0_free_results_dir,
    "Task 0: Free Node",
    outdir,
    outfile="T0_Free_Node_KNN.png",
    operation="attn",
    position=3,
)
create_knn_chart(
    t0_shared_results_dir,
    "Task 0: Shared Node",
    outdir,
    outfile="T0_Shared_Node_KNN.png",
    operation="attn",
    position=3,
)
create_knn_chart(
    t1_free_results_dir,
    "Task 1: Free Node",
    outdir,
    outfile="T1_Free_Node_KNN.png",
    operation="attn",
    position=3,
)
create_knn_chart(
    t1_shared_results_dir,
    "Task 1: Shared Node",
    outdir,
    outfile="T1_Shared_Node_KNN.png",
    operation="attn",
    position=3,
)

create_ablation_chart(
    t0_free_results_dir,
    "Task 0: Free Node",
    outdir,
    outfile="T0_Free_Node_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="",
)
create_ablation_chart(
    t0_free_results_dir,
    "Task 0: Free Node",
    outdir,
    outfile="T0_Free_Node_C_Attn_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_attn",
)
create_ablation_chart(
    t0_free_results_dir,
    "Task 0: Free Node",
    outdir,
    outfile="T0_Free_Node_C_Proj_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_proj",
)

create_ablation_chart(
    t0_shared_results_dir,
    "Task 0: Shared Node",
    outdir,
    outfile="T0_Shared_Node_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="",
)
create_ablation_chart(
    t0_shared_results_dir,
    "Task 0: Shared Node",
    outdir,
    outfile="T0_Shared_Node_C_Attn_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_attn",
)
create_ablation_chart(
    t0_shared_results_dir,
    "Task 0: Shared Node",
    outdir,
    outfile="T0_Shared_Node_C_Proj_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_proj",
)

create_ablation_chart(
    t1_free_results_dir,
    "Task 1: Free Node",
    outdir,
    outfile="T1_Free_Node_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="",
)
create_ablation_chart(
    t1_free_results_dir,
    "Task 1: Free Node",
    outdir,
    outfile="T1_Free_Node_C_Attn_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_attn",
)
create_ablation_chart(
    t1_free_results_dir,
    "Task 1: Free Node",
    outdir,
    outfile="T1_Free_Node_C_Proj_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_proj",
)

create_ablation_chart(
    t1_shared_results_dir,
    "Task 1: Shared Node",
    outdir,
    outfile="T1_Shared_Node_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="",
)
create_ablation_chart(
    t1_shared_results_dir,
    "Task 1: Shared Node",
    outdir,
    outfile="T1_Shared_Node_C_Attn_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_attn",
)
create_ablation_chart(
    t1_shared_results_dir,
    "Task 1: Shared Node",
    outdir,
    outfile="T1_Shared_Node_C_Proj_Ablation.png",
    operation="attn",
    position=3,
    component_prefix="c_proj",
)


t0_free_models, _ = get_models_over_checkpoints(t0_free_results_dir, "attn", 3)
t0_shared_models, epochs = get_models_over_checkpoints(t0_shared_results_dir, "attn", 3)

t0_outdir = os.path.join(outdir, "Task_0_Overlap")
compute_model_overlap(
    t0_free_model_directory,
    t0_shared_model_directory,
    t0_free_models,
    t0_shared_models,
    epochs,
    t0_outdir,
)

t1_free_models, _ = get_models_over_checkpoints(t1_free_results_dir, "attn", 3)
t1_shared_models, epochs = get_models_over_checkpoints(t1_shared_results_dir, "attn", 3)

t1_outdir = os.path.join(outdir, "Task_1_Overlap")
compute_model_overlap(
    t1_free_model_directory,
    t1_shared_model_directory,
    t1_free_models,
    t1_shared_models,
    epochs,
    t1_outdir,
)
