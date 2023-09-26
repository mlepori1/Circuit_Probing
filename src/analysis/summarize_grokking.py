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


    data = pd.DataFrame.from_dict(
        {
            "Epochs": x,
            "Accuracy": y,
            "Method": method
        }
    )
    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
    sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Method").set(title=figtitle)
    plt.savefig(os.path.join(outdir, outfile), format="pdf", bbox_inches="tight")


cp_dir = "../../Results/Probes/Grokking/a2_b/circuit_probing/a2"
linear_dir = "../../Results/Probes/Grokking/a2_b/Probing/Linear/a2"

figtitle = "Probe Accuracy During Training: Attention"
outdir = "./Grokking/"
outfile = "Attention_Probe_Acc.pdf"
create_line_plot(cp_dir, linear_dir, figtitle, outdir, outfile, "attn", sparsity=True)

figtitle = "Probe Accuracy During Training: MLP"
outdir = "./Grokking/"
outfile = "MLP_Probe_Acc.pdf"
create_line_plot(cp_dir, linear_dir, figtitle, outdir, outfile, "mlp", sparsity=True)

cp_dir = "../../Results/Probes/Grokking/a2_b/circuit_probing/b2"
linear_dir = "../../Results/Probes/Grokking/a2_b/Probing/Linear/b2"

figtitle = "Probe Selectivity Analysis"
outdir = "./Grokking/"
outfile = "Attention_Probe_Selectivity.pdf"
create_line_plot(cp_dir, linear_dir, figtitle, outdir, outfile, "attn", sparsity=True)