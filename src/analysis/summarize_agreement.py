import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def plot_zoomed_in(small):
    small = small[(small["operation"] == "attn") & (small["target_layer"] == 6)]
    f = plt.figure(figsize=(5, 3))
    small_df_iid = pd.DataFrame.from_dict(
        {
            "Accuracy": [
                small[" ablated acc IID"].values[0],
                small["random ablate acc mean IID"].values[0],
                small[" vanilla acc IID"].values[0],
            ],
            "Dataset": ["IID", "IID", "IID"],
            "error": [0, small["random ablate acc std IID"].values[0], 0],
            "Condition": ["Ablate Subnetwork", "Ablate Random", "Full Model"],
        }
    )

    small_df_gen = pd.DataFrame.from_dict(
        {
            "Accuracy": [
                small[" ablated acc Gen"].values[0],
                small["random ablate acc mean Gen"].values[0],
                small[" vanilla acc Gen"].values[0],
            ],
            "Dataset": ["OOD", "OOD", "OOD"],
            "error": [0, small["random ablate acc std Gen"].values[0], 0],
            "Condition": ["Ablate Subnetwork", "Ablate Random", "Full Model"],
        }
    )

    small_df = pd.concat([small_df_iid, small_df_gen])

    sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
    with sns.axes_style("darkgrid"):
        g = sns.catplot(
            data=small_df, kind="bar", x="Dataset", y="Accuracy", hue="Condition"
        ).set(title="Subject-Verb Agreement", ylim=(0.5, 1.0))
        for ax in g.axes.flat:
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
            y_coords = [p.get_height() for p in ax.patches]
            x_coords, y_coords = zip(*sorted(zip(x_coords, y_coords)))
            print(x_coords)
            print(y_coords)
            ax.errorbar(
                x=x_coords, y=y_coords, yerr=small_df["error"], fmt="none", c="k"
            )

    f.set_size_inches(3, 5)
    plt.savefig("./Agreement/small_sv_agreement.pdf", format="pdf", bbox_inches="tight")


def plot_everything(df, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df["random ablate acc mean IID"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure()

    plot_df = pd.DataFrame.from_dict(
        {
            "Component": df["Component"],
            "Circuit": df[" ablated acc IID"],
            "Random": df["random ablate acc mean IID"],
        }
    )

    for i in range(0, len(plot_df), 12):
        curr_df = plot_df.iloc[i : i + 12]

        errorbars = []
        errorbars += [0] * len(curr_df["Component"])
        errorbars += list(df["random ablate acc std IID"].values[i : i + 12])

        sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
        ax = sns.catplot(
            data=pd.melt(
                curr_df,
                id_vars="Component",
                var_name="Condition",
                value_name="Ablated Accuracy",
            ),
            kind="bar",
            x="Component",
            y="Ablated Accuracy",
            hue="Condition",
            height=5,
            aspect=6,
        )
        ax._legend.remove()
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.axes[0, 0].patches]
        xmin = ax.axes[0, 0].patches[0].get_x()
        xmax = ax.axes[0, 0].patches[-1].get_x() + ax.axes[0, 0].patches[-1].get_width()
        y_coords = [p.get_height() for p in ax.axes[0, 0].patches]
        ax.axes[0, 0].errorbar(
            x=x_coords, y=y_coords, yerr=errorbars, fmt="none", c="k"
        )
        plt.xticks(rotation=30)
        plt.title(f"GPT2-{figtitle} SV-Agreement Ablation")
        line2 = plt.hlines(
            df[" vanilla acc IID"].values[0],
            xmin=xmin,
            xmax=xmax,
            color="green",
            linestyles="dashed",
            label="Full Acc.",
        )
        # line3 = plt.hlines(df[" vanilla acc Gen"].values[0], xmin=xmin, xmax=xmax, color="blue", linestyles="dashed", label="Gen. Acc.")
        plt.ylim(0.5, 1.0)
        plt.legend(loc="lower right")
        plt.savefig(
            f"./Agreement/{filetitle}_{str(i)}.pdf", format="pdf", bbox_inches="tight"
        )


def plot_knn(df, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df["random ablate acc mean IID"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure(figsize=(30, 10))
    plot_df = pd.DataFrame.from_dict(
        {
            "Component": df["Component"],
            "KNN Test Acc.": df["knn test acc"],
        }
    )
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
    ax = sns.barplot(data=plot_df, x="Component", y="KNN Test Acc.", color="steelblue")
    plt.xticks(rotation=30)
    plt.title(f"GPT2-{figtitle} SV-Agreement KNN Accuracy")
    plt.savefig(f"./Agreement/{filetitle}_SV.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    medium = pd.read_csv("../../Results/Probes/SV_Agr/medium/results.csv")
    small = pd.read_csv("../../Results/Probes/SV_Agr/small/results.csv")
    plot_zoomed_in(small)
    plot_everything(small, "small", "all_small")
    plot_knn(small, "small", "small_knn")
    plot_everything(medium, "medium", "all_medium")
    plot_knn(medium, "medium", "medium_knn")
