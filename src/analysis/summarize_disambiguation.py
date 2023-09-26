import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

cp_a2 = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/circuit_probing/1_Layer/a2/results.csv")
cp_minus_b2 = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/circuit_probing/1_Layer/minus_b2/results.csv")
cp_a_plus_b = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/circuit_probing/1_Layer/a_plus_b/results.csv")
cp_a_minus_b = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/circuit_probing/1_Layer/a_minus_b/results.csv")

linear_a2 = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Linear/1_Layer/a2/results.csv")
linear_minus_b2 = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Linear/1_Layer/minus_b2/results.csv")
linear_a_plus_b = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Linear/1_Layer/a_plus_b/results.csv")
linear_a_minus_b = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Linear/1_Layer/a_minus_b/results.csv")

nonlinear_a2 = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Nonlinear/1_Layer/a2/results.csv")
nonlinear_minus_b2 = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Nonlinear/1_Layer/minus_b2/results.csv")
nonlinear_a_plus_b = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Nonlinear/1_Layer/a_plus_b/results.csv")
nonlinear_a_minus_b = pd.read_csv("../../Results/Probes/Disambiguation/a2_minus_b2/Probing/Nonlinear/1_Layer/a_minus_b/results.csv")


### Create Bar Graph
def subsample_cp_result(df, var):
    df_bar = df[["operation", "knn test acc"]]
    df_bar["Method"] = ["Circuit Probe"] * 2
    df_bar["Variable"] = [var] * 2
    df_bar = df_bar.rename(columns={"operation": "Component", "knn test acc": "Test Accuracy"})
    return df_bar

cp_a2_bar = subsample_cp_result(cp_a2, "a2")
cp_minus_b2_bar = subsample_cp_result(cp_minus_b2, "-b2")
cp_a_plus_b_bar = subsample_cp_result(cp_a_plus_b, "a+b")
cp_a_minus_b_bar = subsample_cp_result(cp_a_minus_b, "a-b")

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

linear_a2_bar = subsample_probing_result(linear_a2, "a2", "Linear Probe")
linear_minus_b2_bar = subsample_probing_result(linear_minus_b2, "-b2", "Linear Probe")
linear_a_plus_b_bar = subsample_probing_result(linear_a_plus_b, "a+b", "Linear Probe")
linear_a_minus_b_bar = subsample_probing_result(linear_a_minus_b, "a-b", "Linear Probe")

nonlinear_a2_bar = subsample_probing_result(nonlinear_a2, "a2", "Nonlinear Probe")
nonlinear_minus_b2_bar = subsample_probing_result(nonlinear_minus_b2, "-b2", "Nonlinear Probe")
nonlinear_a_plus_b_bar = subsample_probing_result(nonlinear_a_plus_b, "a+b", "Nonlinear Probe")
nonlinear_a_minus_b_bar = subsample_probing_result(nonlinear_a_minus_b, "a-b", "Nonlinear Probe")

data = pd.concat([cp_a2_bar, cp_minus_b2_bar, cp_a_plus_b_bar, cp_a_minus_b_bar, linear_a2_bar, linear_minus_b2_bar, linear_a_plus_b_bar, linear_a_minus_b_bar, nonlinear_a2_bar, nonlinear_minus_b2_bar, nonlinear_a_plus_b_bar, nonlinear_a_minus_b_bar])

plt.figure()
sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
sns.catplot(data=data[data["Component"] == "attn"], kind="bar", x="Variable", y="Test Accuracy", hue="Method").set(title="Probe Test Accuracy: Attention")
plt.savefig("./Disambiguation/Attention_Probe_Acc.pdf", format="pdf", bbox_inches="tight")

plt.figure()
sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
sns.catplot(data=data[data["Component"] == "mlp"], kind="bar", x="Variable", y="Test Accuracy", hue="Method").set(title="Probe Test Accuracy: MLP")
plt.savefig("./Disambiguation/MLP_Probe_Acc.pdf", format="pdf", bbox_inches="tight")