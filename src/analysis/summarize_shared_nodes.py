import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

def create_bar_chart(task, title):
    cp_free = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/circuit_probing/free/results.csv")
    cp_shared = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/circuit_probing/shared/results.csv")
    cp_other = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/circuit_probing/aux/results.csv")

    linear_free = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/Probing/Linear/free/results.csv")
    linear_shared = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/Probing/Linear/shared/results.csv")
    linear_other = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/Probing/Linear/aux/results.csv")

    nonlinear_free = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/Probing/Nonlinear/free/results.csv")
    nonlinear_shared = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/Probing/Nonlinear/shared/results.csv")
    nonlinear_other = pd.read_csv(f"../../Results/Probes/Shared_Nodes/{task}/Probing/Nonlinear/aux/results.csv")

    ### Create Bar Graph
    def subsample_cp_result(df, var):
        df_bar = df[["operation", "knn test acc"]]
        df_bar["Method"] = ["Circuit Probe"] * 2
        df_bar["Variable"] = [var] * 2
        df_bar = df_bar.rename(columns={"operation": "Component", "knn test acc": "Test Accuracy"})
        return df_bar

    cp_free = subsample_cp_result(cp_free, "Free")
    cp_shared = subsample_cp_result(cp_shared, "Shared")
    cp_other = subsample_cp_result(cp_other, "Other")

    def subsample_probing_result(df, var, probe_str):
        df_bar = df[["residual_location", "test acc"]]
        df_bar["Method"] = [probe_str] * 2
        df_bar["Variable"] = [var] * 2
        df_bar = df_bar.rename(columns={"residual_location": "Component", "test acc": "Test Accuracy"})
        df_bar["Component"] = df_bar["Component"].replace({
            "resid_mid": "attn",
            "resid_post": "mlp"
        })
        return df_bar

    linear_free = subsample_probing_result(linear_free, "Free", "Linear Probe")
    linear_shared = subsample_probing_result(linear_shared, "Shared", "Linear Probe")
    linear_other = subsample_probing_result(linear_other, "Other", "Linear Probe")

    nonlinear_free = subsample_probing_result(nonlinear_free, "Free", "Nonlinear Probe")
    nonlinear_shared = subsample_probing_result(nonlinear_shared, "Shared", "Nonlinear Probe")
    nonlinear_other = subsample_probing_result(nonlinear_other, "Other", "Nonlinear Probe")

    data = pd.concat([cp_free, cp_shared, cp_other, linear_free, linear_shared, linear_other, nonlinear_free, nonlinear_shared, nonlinear_other])

    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
    sns.catplot(data=data[data["Component"] == "attn"], kind="bar", x="Variable", y="Test Accuracy", hue="Method").set(title=title + ": Attention")
    plt.savefig(f"./Shared_Node/{task}_Attention_Probe_Acc.pdf", format="pdf", bbox_inches="tight")

    plt.figure()
    sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
    sns.catplot(data=data[data["Component"] == "mlp"], kind="bar", x="Variable", y="Test Accuracy", hue="Method").set(title=title + ": MLP")
    plt.savefig(f"./Shared_Node/{task}_MLP_Probe_Acc.pdf", format="pdf", bbox_inches="tight")


create_bar_chart("Task_1", "Task 1 Probe Accuracy")
create_bar_chart("Task_2", "Task 2 Probe Accuracy")