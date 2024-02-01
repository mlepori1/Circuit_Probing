import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


### Create Line plot with both linear probe and circuit probe performance
def create_line_plot(
    cp_dir,
    linear_dir,
    nonlinear_dir,
    figtitle,
    outdir,
    outfile,
    operation,
    sparsity=False,
):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(cp_dir)

    cp_y = []
    linear_y = []
    nonlinear_y = []
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

        data = pd.read_csv(os.path.join(nonlinear_dir, str(ch), "results.csv"))
        data = data[data["residual_location"] == resid]
        nonlinear_y.append(data["test acc"].values[0])

    y = cp_y + linear_y + nonlinear_y
    method = (
        ["Circuit Probe"] * len(cp_y)
        + ["Linear Probe"] * len(linear_y)
        + ["Nonlinear Probe"] * len(nonlinear_y)
    )
    x = x + x + x

    data = pd.DataFrame.from_dict({"Epochs": x, "Accuracy": y, "Method": method})
    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.75)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Method").set(title=figtitle)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


### Create Line plot with both circuit probing ablation
def create_cp_ablation_plot(
    a2_dir,
    b2_dir,
    figtitle,
    outdir,
    outfile,
    operation,
):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(cp_dir)

    a_y = []
    b_y = []
    x = [int(ch) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(a2_dir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        a_y.append(data[" ablated acc Test"].values[0])

        data = pd.read_csv(os.path.join(b2_dir, str(ch), "results.csv"))
        data = data[data["operation"] == operation]
        b_y.append(data[" ablated acc Test"].values[0])

    y = a_y + b_y
    variable = ["a2"] * len(a_y) + ["b2"] * len(b_y)
    x = x + x

    data = pd.DataFrame.from_dict({"Epochs": x, "Accuracy": y, "Var.": variable})
    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.75)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Var.").set(title=figtitle)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


### Create Amnesic probing plot
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
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.75)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Var.").set(title=figtitle)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


### Create DAS plot
def create_DAS_plot(a2, b2, figtitle, outdir, outfile, operation):
    os.makedirs(outdir, exist_ok=True)
    checkpoints = os.listdir(a2)

    y1 = []
    y2 = []
    x = [int(ch.split("_")[0]) for ch in checkpoints]
    x.sort()

    for ch in x:
        data = pd.read_csv(os.path.join(a2, str(ch) + f"_{operation}", "eval_log.txt"))
        y1.append(data["accuracy"].values[-1])

        data = pd.read_csv(os.path.join(b2, str(ch) + f"_{operation}", "eval_log.txt"))
        y2.append(data["accuracy"].values[-1])

    y = y1 + y2
    var = ["a2"] * len(y1) + ["b2"] * len(y2)
    x = x + x

    data = pd.DataFrame.from_dict({"Epochs": x, "Accuracy": y, "Var.": var})
    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.75)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Var.").set(title=figtitle)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


cp_dir = "../../Results/Probes/Grokking/circuit_probing/a2"
linear_dir = "../../Results/Probes/Grokking/Probing/Linear/a2"
nonlinear_dir = "../../Results/Probes/Grokking/Probing/Nonlinear/a2"

figtitle = "Probe Accuracy During Training: Attn."
outdir = "./Grokking/"
outfile = "Attention_Probe_Acc.pdf"
create_line_plot(
    cp_dir, linear_dir, nonlinear_dir, figtitle, outdir, outfile, "attn", sparsity=True
)

figtitle = "Probe Accuracy During Training: MLP"
outdir = "./Grokking/"
outfile = "MLP_Probe_Acc.pdf"
create_line_plot(
    cp_dir, linear_dir, nonlinear_dir, figtitle, outdir, outfile, "mlp", sparsity=True
)

cp_dir = "../../Results/Probes/Grokking/circuit_probing/b2"
linear_dir = "../../Results/Probes/Grokking/Probing/Linear/b2"
nonlinear_dir = "../../Results/Probes/Grokking/Probing/Nonlinear/b2"

figtitle = "Probe Selectivity Analysis"
outdir = "./Grokking/"
outfile = "Attention_Probe_Selectivity.pdf"
create_line_plot(
    cp_dir, linear_dir, nonlinear_dir, figtitle, outdir, outfile, "attn", sparsity=True
)

a2_dir = "../../Results/Probes/Grokking/Amnesic/a2"
b2_dir = "../../Results/Probes/Grokking/Amnesic/b2"

figtitle = "Amnesic Probing Analysis: Attn."
outdir = "./Grokking/"
outfile = "Amnesic_attn.pdf"
create_amnesic_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "attn")

figtitle = "Amnesic Probing Analysis: MLP"
outfile = "Amnesic_mlp.pdf"
create_amnesic_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "mlp")

a2_dir = "../../Results/Probes/Grokking/DAS/a2"
b2_dir = "../../Results/Probes/Grokking/DAS/b2"

figtitle = "Causal Abstraction Analysis: Attn."
outdir = "./Grokking/"
outfile = "das_attn.pdf"
create_DAS_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "attn")

figtitle = "Causal Abstraction Analysis: MLP"
outfile = "das_mlp.pdf"
create_DAS_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "mlp")

a2_dir = "../../Results/Probes/Grokking/circuit_probing/a2"
b2_dir = "../../Results/Probes/Grokking/circuit_probing/b2"
outdir = "./Grokking/"
outfile = "cp_ablation_attn.pdf"
figtitle = "Circuit Probing Ablation: Attn."
create_cp_ablation_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "attn")

outfile = "cp_ablation_mlp.pdf"
figtitle = "Circuit Probing Ablation: MLP."
create_cp_ablation_plot(a2_dir, b2_dir, figtitle, outdir, outfile, "mlp")
