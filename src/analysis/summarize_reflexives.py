import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def plot_zoomed_in(small, medium):
  medium = medium[(medium["operation"] == "mlp") & (medium["target_layer"] == 0)]
  small = small[(small["operation"] == "mlp") & (small["target_layer"] == 0)]

  plt.figure()
  medium_df = pd.DataFrame.from_dict({
      "Ablated Accuracy": [medium[" ablated acc Male IID"].values[0], 
                          medium["random ablate acc mean Male IID"].values[0], 
                          medium[" ablated acc Male Gen"].values[0], 
                          medium["random ablate acc mean Male Gen"].values[0],
                          ],
      "Condition": ["Sub.: IID", 
                    "Rand.: IID", 
                    "Sub.: Gen.",
                      "Rand.: Gen.",
                      ],
      "error": [0, 
                medium["random ablate acc std Male IID"].values[0],
                  0,  
                  medium["random ablate acc std Male Gen"].values[0],
                  ]
  } )

  sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
  ax = sns.barplot(data=medium_df, x="Condition", y="Ablated Accuracy")
  print(ax)
  x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
  xmin = ax.patches[0].get_x()
  xmax = ax.patches[3].get_x() + ax.patches[3].get_width()
  y_coords = [p.get_height() for p in ax.patches]
  ax.errorbar(x=x_coords, y=y_coords, yerr=medium_df["error"], fmt="none", c="k")
  plt.xticks(rotation=30)
  plt.title("GPT2-Medium Male Reflexives Ablation: Layer 0 MLP")
  plt.hlines(.5, xmin=xmin, xmax=xmax, color="red", linestyles="dotted", label="Chance")
  plt.hlines(medium[" vanilla acc Male IID"], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="M IID Acc.")
  plt.hlines(medium[" vanilla acc Male Gen"], xmin=xmin, xmax=xmax, color="blue", linestyles="dashed", label="M Gen. Acc.")
  plt.legend(loc="lower right")
  plt.savefig("./Reflexive_An/medium_male_reflexives.pdf", format="pdf", bbox_inches="tight")


  plt.figure()
  medium_df = pd.DataFrame.from_dict({
      "Ablated Accuracy": [
                          medium[" ablated acc Female IID"].values[0], 
                          medium["random ablate acc mean Female IID"].values[0], 
                          medium[" ablated acc Female Gen"].values[0], 
                          medium["random ablate acc mean Female Gen"].values[0],
                          ],
      "Condition": [
                      "Sub.: IID", 
                    "Rand.: IID", 
                    "Sub.: Gen.",
                      "Rand.: Gen.",
                      ],
      "error": [
                  0,
                medium["random ablate acc std Female IID"].values[0],
                  0,
                  medium["random ablate acc std Female Gen"].values[0],
                  ]
  } )

  sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
  ax = sns.barplot(data=medium_df, x="Condition", y="Ablated Accuracy")
  print(ax)
  x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
  xmin = ax.patches[0].get_x()
  xmax = ax.patches[3].get_x() + ax.patches[3].get_width()
  y_coords = [p.get_height() for p in ax.patches]
  ax.errorbar(x=x_coords, y=y_coords, yerr=medium_df["error"], fmt="none", c="k")
  plt.xticks(rotation=30)
  plt.title("GPT2-Medium Female Reflexives Ablation: Layer 0 MLP")
  plt.hlines(.5, xmin=xmin, xmax=xmax, color="red", linestyles="dotted", label="Chance")
  plt.hlines(medium[" vanilla acc Female IID"], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="F IID Acc.")
  plt.hlines(medium[" vanilla acc Female Gen"], xmin=xmin, xmax=xmax, color="blue", linestyles="dashed", label="F Gen. Acc.")
  plt.legend(loc="lower right")
  plt.savefig("./Reflexive_An/medium_female_reflexives.pdf", format="pdf", bbox_inches="tight")


  plt.figure()
  small_df = pd.DataFrame.from_dict({
      "Ablated Accuracy": [small[" ablated acc Male IID"].values[0], 
                          small["random ablate acc mean Male IID"].values[0], 
                          small[" ablated acc Male Gen"].values[0], 
                          small["random ablate acc mean Male Gen"].values[0],
                          ],
      "Condition": ["Sub.: IID", 
                    "Rand.: IID", 
                    "Sub.: Gen.",
                      "Rand.: Gen.",
                      ],
      "error": [0, 
                small["random ablate acc std Male IID"].values[0],
                  0,  
                  small["random ablate acc std Male Gen"].values[0],
                  ]
  } )

  sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
  ax = sns.barplot(data=small_df, x="Condition", y="Ablated Accuracy")
  print(ax)
  x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
  xmin = ax.patches[0].get_x()
  xmax = ax.patches[3].get_x() + ax.patches[3].get_width()
  y_coords = [p.get_height() for p in ax.patches]
  ax.errorbar(x=x_coords, y=y_coords, yerr=small_df["error"], fmt="none", c="k")
  plt.xticks(rotation=30)
  plt.title("GPT2-Small Male Reflexives Ablation: Layer 0 MLP")
  plt.hlines(.5, xmin=xmin, xmax=xmax, color="red", linestyles="dotted", label="Chance")
  plt.hlines(small[" vanilla acc Male IID"], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="M IID Acc.")
  plt.hlines(small[" vanilla acc Male Gen"], xmin=xmin, xmax=xmax, color="blue", linestyles="dashed", label="M Gen. Acc.")
  plt.legend(loc="lower right")
  plt.savefig("./Reflexive_An/small_male_reflexives.pdf", format="pdf", bbox_inches="tight")



  plt.figure()
  small_df = pd.DataFrame.from_dict({
      "Ablated Accuracy": [
                          small[" ablated acc Female IID"].values[0], 
                          small["random ablate acc mean Female IID"].values[0], 
                          small[" ablated acc Female Gen"].values[0], 
                          small["random ablate acc mean Female Gen"].values[0],
                          ],
      "Condition": [
                      "Sub.: IID", 
                    "Rand.: IID", 
                    "Sub.: Gen.",
                      "Rand.: Gen.",
                      ],
      "error": [
                  0,
                small["random ablate acc std Female IID"].values[0],
                  0,
                  small["random ablate acc std Female Gen"].values[0],
                  ]
  } )
  sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
  ax = sns.barplot(data=small_df, x="Condition", y="Ablated Accuracy")
  print(ax)
  x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
  xmin = ax.patches[0].get_x()
  xmax = ax.patches[3].get_x() + ax.patches[3].get_width()
  y_coords = [p.get_height() for p in ax.patches]
  ax.errorbar(x=x_coords, y=y_coords, yerr=small_df["error"], fmt="none", c="k")
  plt.xticks(rotation=30)
  plt.title("GPT2-Small Female Reflexives Ablation: Layer 0 MLP")
  plt.hlines(.5, xmin=xmin, xmax=xmax, color="red", linestyles="dotted", label="Chance")
  plt.hlines(small[" vanilla acc Female IID"], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="F IID Acc.")
  plt.hlines(small[" vanilla acc Female Gen"], xmin=xmin, xmax=xmax, color="blue", linestyles="dashed", label="F Gen. Acc.")
  plt.legend(loc="lower right")
  plt.savefig("./Reflexive_An/small_female_reflexives.pdf", format="pdf", bbox_inches="tight")


def plot_everything(df, gender, figtitle, filetitle):
    # Get rid of data points whose subnetworks comprise > 50% of any tensor
    df = df[df[f"random ablate acc mean {gender} IID"] != -1]
    df["Component"] = df["operation"] + "-" + df["target_layer"].astype(str)
    plt.figure()
    plot_df = pd.DataFrame.from_dict({
        "Component": df["Component"],
        "Sub.: IID": df[f" ablated acc {gender} IID"],
        "Rand.: IID": df[f"random ablate acc mean {gender} IID"],
        "Sub.: Gen:": df[f" ablated acc {gender} Gen"],
        "Rand. : Gen": df[f"random ablate acc mean {gender} Gen"],
    } )

    for i in range(0, len(plot_df), 12):
        curr_df = plot_df.iloc[i:i+12]
        errorbars = []
        errorbars += [0] * len(curr_df["Component"])
        errorbars += list(df[f"random ablate acc std {gender} IID"].values[i:i+12])
        errorbars += [0] * len(curr_df["Component"])
        errorbars += list(df[f"random ablate acc std {gender} Gen"].values[i:i+12])

        sns.set(style="darkgrid", palette="Dark2", font_scale=1.5)
        ax = sns.catplot(data=pd.melt(curr_df, id_vars="Component", var_name="Condition", value_name="Ablated Accuracy"), kind="bar", x="Component", y="Ablated Accuracy", hue="Condition", height=5, aspect=6)
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.axes[0,0].patches]
        xmin = ax.axes[0,0].patches[0].get_x()
        xmax = ax.axes[0,0].patches[-1].get_x() + ax.axes[0,0].patches[-1].get_width()
        y_coords = [p.get_height() for p in ax.axes[0,0].patches]
        ax.axes[0,0].errorbar(x=x_coords, y=y_coords, yerr=errorbars, fmt="none", c="k")
        plt.xticks(rotation=30)
        plt.title(f"GPT2-{figtitle} {gender} Reflexives Ablation")
        line1 = plt.hlines(.5, xmin=xmin, xmax=xmax, color="red", linestyles="dotted", label="Chance")
        line2 = plt.hlines(df[f" vanilla acc {gender} IID"].values[0], xmin=xmin, xmax=xmax, color="green", linestyles="dashed", label="IID Acc.")
        line3 = plt.hlines(df[f" vanilla acc {gender} Gen"].values[0], xmin=xmin, xmax=xmax, color="blue", linestyles="dashed", label="Gen. Acc.")
        plt.legend((line1, line2, line3), ("Chance", "IID Acc.", "Gen. Acc."), loc="lower right")
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
    plt.savefig(f"./Reflexive_An/{filetitle}.pdf", format="pdf", bbox_inches="tight")



if __name__=="__main__":
  medium = pd.read_csv("../../Results/Probes/Reflexive_An/medium/results.csv")
  small = pd.read_csv("../../Results/Probes/Reflexive_An/small/results.csv")
  plot_zoomed_in(small, medium)
  plot_everything(small, "Male", "small", "all_male_small")
  plot_everything(medium, "Male", "medium", "all_male_medium")
  plot_everything(small, "Female", "small", "all_female_small")
  plot_everything(medium, "Female", "medium", "all_female_medium")

  plot_knn(small, "small", "small_knn")
  plot_knn(medium, "medium", "medium_knn")

  
