import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def plot_zoomed_in(df, size, gender, figtitle, layer):
    df = df[(df["operation"] == "attn") & (df["target_layer"] == layer)]

    f = plt.figure()
    df_iid = pd.DataFrame.from_dict({
        "Ablated Accuracy": [df[f" ablated acc {gender} IID"].values[0], df[f"random ablate acc mean {gender} IID"].values[0]],
        "Condition": ["Abl. Circuit", "Abl. Random"],
        "error": [0, df[f"random ablate acc std {gender} IID"].values[0]]
    } )

    sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
    with sns.axes_style("darkgrid"):
        ax0 = sns.barplot(data=df_iid, x="Condition", y="Ablated Accuracy")

        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax0.patches]
        xmin = ax0.patches[0].get_x()
        xmax = ax0.patches[-1].get_x() + ax0.patches[-1].get_width()
        y_coords = [p.get_height() for p in ax0.patches]
        ax0.errorbar(x=x_coords, y=y_coords, yerr=df_iid["error"], fmt="none", c="k")
        ax0.hlines(df[f" vanilla acc {gender} IID"], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="Full Model Acc.")

        ax0.tick_params(axis='x',labelrotation=30, labelsize=15)
        ax0.set_ylim(0.5, 1.0)
        ax0.set_ylabel(ylabel="Ablated Accuracy", fontsize=15)
        ax0.set_xlabel(xlabel="Condition", fontsize=15)

        f.suptitle(f"GPT2 {figtitle} RA: Attn-{str(layer)}")
        f.set_size_inches(3, 5)
        plt.legend(loc="lower right")
        plt.savefig(f"./Reflexive_An/{size}_{gender}_reflexives.pdf", format="pdf", bbox_inches="tight")

def plot_everything(df, gender, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df[f"random ablate acc mean {gender} IID"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure()
    plot_df = pd.DataFrame.from_dict({
        "Component": df["Component"],
        "Circuit": df[f" ablated acc {gender} IID"],
        "Random": df[f"random ablate acc mean {gender} IID"],
    } )

    for i in range(0, len(plot_df), 12):
        curr_df = plot_df.iloc[i:i+12]
        errorbars = []
        errorbars += [0] * len(curr_df["Component"])
        errorbars += list(df[f"random ablate acc std {gender} IID"].values[i:i+12])

        sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
        ax = sns.catplot(data=pd.melt(curr_df, id_vars="Component", var_name="Condition", value_name="Ablated Accuracy"), kind="bar", x="Component", y="Ablated Accuracy", hue="Condition", height=5, aspect=6)
        ax._legend.remove()
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.axes[0,0].patches]
        xmin = ax.axes[0,0].patches[0].get_x()
        xmax = ax.axes[0,0].patches[-1].get_x() + ax.axes[0,0].patches[-1].get_width()
        y_coords = [p.get_height() for p in ax.axes[0,0].patches]
        ax.axes[0,0].errorbar(x=x_coords, y=y_coords, yerr=errorbars, fmt="none", c="k")
        plt.xticks(rotation=30)
        plt.title(f"GPT2-{figtitle} {gender} Reflexives Ablation")
        line2 = plt.hlines(df[f" vanilla acc {gender} IID"].values[0], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="IID Acc.")
        plt.legend(loc="lower right")
        plt.savefig(f"./Reflexive_An/{filetitle}_{str(i)}.pdf", format="pdf", bbox_inches="tight")


def plot_knn(df, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df["random ablate acc mean Female IID"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure(figsize=(30, 10))
    plot_df = pd.DataFrame.from_dict({
        "Component": df["Component"],
        "KNN Test Acc.": df["knn test acc"],
    } )
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
    ax = sns.barplot(data=plot_df, x="Component", y="KNN Test Acc.", color="steelblue")
    plt.xticks(rotation=30)
    plt.title(f"GPT2-{figtitle} Reflexive KNN Accuracy")
    plt.savefig(f"./Reflexive_An/{filetitle}_RA.pdf", format="pdf", bbox_inches="tight")



if __name__=="__main__":
  medium = pd.read_csv("../../Results/Probes/Reflexive_An_Mixed/medium/results.csv")
  small = pd.read_csv("../../Results/Probes/Reflexive_An_Mixed/small/results.csv")
  plot_zoomed_in(small, "small", "Male", "M", 6)
  plot_zoomed_in(small, "small", "Female", "F", 6)
  plot_zoomed_in(medium, "medium", "Male", "-med. M", 7)
  plot_zoomed_in(medium, "medium", "Female", "-med. F", 7)

  plot_everything(small, "Male", "small", "all_male_small")
  plot_everything(medium, "Male", "medium", "all_male_medium")
  plot_everything(small, "Female", "small", "all_female_small")
  plot_everything(medium, "Female", "medium", "all_female_medium")

  plot_knn(small, "small", "small_knn")
  plot_knn(medium, "medium", "medium_knn")

  
