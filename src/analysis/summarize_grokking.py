import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


### Create Line plot with both linear probe and circuit probe performance
def create_line_plot(
    cp_dir, linear_dir, figtitle, outdir, outfile, operation, sparsity=False
):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(cp_dir)

    cp_y = []
    linear_y = []
    sparsity = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(cp_dir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        cp_y.append(data["knn test acc"].values[0])
        sparsity.append(data["L0 Norm"].values[0] / data["L0 Max"].values[0])

        data = pd.read_csv(os.path.join(linear_dir, str(ch), "results.csv"))
        if operation == "attn":
            resid = "resid_mid"
        else:
            resid = "resid_post"
        data = data[data["residual_location"] == resid]
        linear_y.append(data["test acc"].values[0])

    y = cp_y + linear_y
    method = ["Circuit Probe"] * len(cp_y) + ["Linear Probe"] * len(linear_y)
    x = x + x

    data = pd.DataFrame.from_dict({"Epochs": x, "Accuracy": y, "Method": method})
    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Method").set(title=figtitle)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


### Create Amnesic probingplot
def create_amnesic_plot(a2, b2, figtitle, outdir, outfile, operation):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(a2)

    y1 = []
    y2 = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    if operation == "attn":
        resid = "resid_mid"
    else:
        resid = "resid_post"

    for ch in x:
        data = pd.read_csv(os.path.join(a2, str(ch), "results.csv"))
        data = data[data["residual_location"] == resid]
        y1.append(data["test acc"].values[0])

        data = pd.read_csv(os.path.join(b2, str(ch), "results.csv"))
        data = data[data["residual_location"] == resid]
        y2.append(data["test acc"].values[0])

    y = y1 + y2
    var = ["a2"] * len(y1) + ["b2"] * len(y2)
    x = x + x

    data = pd.DataFrame.from_dict({"Epochs": x, "Accuracy": y, "Var.": var})
    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Var.").set(title=figtitle)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


cp_dir = "../../Results/Probes/Grokking/circuit_probing/a2"
linear_dir = "../../Results/Probes/Grokking/Probing/Linear/a2"

figtitle = "Probe Accuracy During Training: Attention"
outdir = "./Grokking/"
outfile = "Attention_Probe_Acc.pdf"
create_line_plot(cp_dir, linear_dir, figtitle, outdir, outfile, "attn", sparsity=True)

figtitle = "Probe Accuracy During Training: MLP"
outdir = "./Grokking/"
outfile = "MLP_Probe_Acc.pdf"
create_line_plot(cp_dir, linear_dir, figtitle, outdir, outfile, "mlp", sparsity=True)

cp_dir = "../../Results/Probes/Grokking/circuit_probing/b2"
linear_dir = "../../Results/Probes/Grokking/Probing/Linear/b2"

figtitle = "Probe Selectivity Analysis"
outdir = "./Grokking/"
outfile = "Attention_Probe_Selectivity.pdf"
create_line_plot(cp_dir, linear_dir, figtitle, outdir, outfile, "attn", sparsity=True)

a2_dir = "../../Results/Probes/Grokking/Amnesic/a2"
b2_dir = "../../Results/Probes/Grokking/Amnesic/b2"

figtitle = "Amnesic Probing Analysis: Attention"
outdir = "./Grokking/"
outfile = "Amnesic_attn.pdf"
create_amnesic_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "attn")

figtitle = "Amnesic Probing Analysis: MLP"
outfile = "Amnesic_mlp.pdf"
create_amnesic_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "mlp")
