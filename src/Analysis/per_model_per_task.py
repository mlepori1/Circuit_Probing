import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dir",
        default=None,
        help="where to load results csv",
        metavar="FILE",
    )

    argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)

    if not hasattr(args, "dir"):
        raise ValueError("Must include path to results file")
    suffix = os.path.split(args.dir)[-1].lower()
    
    probe_results = pd.read_csv(os.path.join(args.dir, suffix, 'results.csv'))
    froze_model_results = pd.read_csv(os.path.join(args.dir, f'froz-{suffix}', 'results.csv'))
    trained_random_results = pd.read_csv(os.path.join(args.dir, f'rand-{suffix}', 'results.csv'))
    embeds_results = pd.read_csv(os.path.join(args.dir, f'rand-embeds-{suffix}', 'results.csv'))
    froze_random_results = pd.read_csv(os.path.join(args.dir, f'rand-froz-{suffix}', 'results.csv'))
    layer_results = pd.read_csv(os.path.join(args.dir, f'rand-layer-{suffix}', 'results.csv'))

    results = {
        "probe": probe_results,
        "frozen": froze_model_results,
        "trained_random": trained_random_results,
        "embeddings": embeds_results,
        "frozen_random": froze_random_results,
        "layer": layer_results
    }

    # Set up directory
    path, task = os.path.split(args.dir)
    _, model = os.path.split(path)
    outdir = os.path.join("../Figs", model, task)
    os.makedirs(outdir, exist_ok=True)

    # Per layer, plot probe Dev and Test accuracy, and plot performance of every other configuration for comparison
    title = f'{model}-{task} KNN Accuracy Per Layer'
    fig = plt.figure(figsize=(30, 10))
    n_layers = len(results["probe"]["model_id"])
    base_xs = np.array((range(0, int(n_layers * 6), 6)))
    x_labels = list(range(n_layers))
    
    plt.bar(base_xs, results["probe"]["knn dev acc"], width=.5, color="blue", edgecolor="black", label="Probe-Dev")
    plt.bar(base_xs + .5, results["layer"]["knn dev acc"], width=.5, color="blue", hatch="O", edgecolor="black", label="Layer-Dev")
    plt.bar(base_xs + 1, results["frozen"]["knn dev acc"], width=.5, color="blue", hatch="/", edgecolor="black", label="Frozen-Dev")
    plt.bar(base_xs + 1.5, results["embeddings"]["knn dev acc"], width=.5, color="blue", hatch="x", edgecolor="black", label="Embeddings-Dev")
    plt.bar(base_xs + 2, results["trained_random"]["knn dev acc"], width=.5, color="blue", hatch="*", edgecolor="black", label="Trained-Random-Dev")
    plt.bar(base_xs + 2.5, results["frozen_random"]["knn dev acc"], width=.5, color="blue", hatch="o", edgecolor="black", label="Frozen-Random-Dev")

    plt.bar(base_xs + 3, results["probe"]["knn test acc"], width=.5, color="red", edgecolor="black", label="Probe-Test")
    plt.bar(base_xs + 3.5, results["layer"]["knn test acc"], width=.5, color="red", hatch="O", edgecolor="black", label="Probe-Test")
    plt.bar(base_xs + 4, results["frozen"]["knn test acc"], width=.5, color="red", hatch="/", edgecolor="black", label="Frozen-Test")
    plt.bar(base_xs + 4.5, results["embeddings"]["knn test acc"], width=.5, color="red", hatch="x", edgecolor="black", label="Embeddings-Test")
    plt.bar(base_xs + 5, results["trained_random"]["knn test acc"], width=.5, color="red", hatch="*", edgecolor="black", label="Trained-Random-Test")
    plt.bar(base_xs + 5.5, results["frozen_random"]["knn test acc"], width=.5, color="red", hatch="o", edgecolor="black", label="Frozen-Random-Test")

    plt.axhline(y=results["probe"]["dev majority acc"].iloc[0].item(), color="b", linestyle="--")
    plt.axhline(y=results["probe"]["test majority acc"].iloc[0].item(), color="r", linestyle="--")
    plt.xticks(base_xs + 2.5, labels = x_labels)
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("KNN Accuracy")
    plt.legend()
    plt.savefig(os.path.join(outdir, "Per_Layer_KNN.png"))

    # Per layer, plot acc difference when ablating cs acc difference with random ablation
    title = f'{model}-{task} LM Accuracy Difference Per Layer'
    if results["probe"]["mlm vanilla acc"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(30, 20))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(30, 20))

    fig.suptitle(title, fontsize=35)
    n_layers = len(results["probe"]["model_id"])
    base_xs = np.array((range(0, int(n_layers * 3), 3)))/2
    x_labels = list(range(n_layers))

    probe_diffs = results["probe"]["vanilla acc"] - results["probe"]["ablated acc"]
    random_diffs = results["probe"]["vanilla acc"] - results["probe"]["random ablate acc"]

    axs[0].bar(base_xs, probe_diffs, width=.5, color="blue", edgecolor="black", label="Ablate Probe Subnetwork")
    axs[0].bar(base_xs+.5, random_diffs, width=.5, color="r", edgecolor="black", label="Ablate Random Subnetwork")

    axs[0].set_xticks(base_xs + .5, labels = x_labels, size=20)
    axs[0].set_ylim(-0.1, 1.0)
    axs[0].tick_params(labelsize=20)
    axs[0].set_title("Language Modeling", size=30)
    axs[0].set_xlabel("Layer", size=20)
    axs[0].set_ylabel("LM Top-1 Accuracy Difference (Full Acc - Ablated Acc)", size=20)
    axs[0].axhline(0, color="black", linestyle="--")

    if incl_mlm:
        probe_diffs = results["probe"]["mlm vanilla acc"] - results["probe"]["mlm ablated acc"]
        random_diffs = results["probe"]["mlm vanilla acc"] - results["probe"]["random ablate mlm acc"]

        axs[1].bar(base_xs, probe_diffs, width=.5, color="blue", edgecolor="black", label="Ablate Probe Subnetwork")
        axs[1].bar(base_xs+.5, random_diffs, width=.5, color="r", edgecolor="black", label="Ablate Random Subnetwork")

        axs[1].set_xticks(base_xs + .5, labels = x_labels, size=20)
        axs[1].set_ylim(-0.1, 1.0)
        axs[1].tick_params(labelsize=20)
        axs[1].set_title("Masked Language Modeling", size=30)
        axs[1].set_xlabel("Layer", size=20)
        axs[1].set_ylabel("LM Top-1 Accuracy Difference (Full Acc - Ablated Acc)", size=20)
        axs[1].axhline(0, color="black", linestyle="--")

    axs[0].legend(fontsize=25)
    plt.savefig(os.path.join(outdir, "Per_Layer_LM_Acc_Diffs.png"))

    # Per layer, plot KL when ablating vs. KL with random ablation
    title = f'{model}-{task} LM KL Per Layer'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(30, 20))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(30, 20))

    fig.suptitle(title, fontsize=35)
    n_layers = len(results["probe"]["model_id"])
    base_xs = np.array((range(0, int(n_layers * 3), 3)))/2
    x_labels = list(range(n_layers))

    probe_kl = results["probe"]["kl"]
    random_kl = results["probe"]["random ablate kl"]

    axs[0].bar(base_xs, probe_kl, width=.5, color="blue", edgecolor="black", label="Ablate Probe Subnetwork")
    axs[0].bar(base_xs+.5, random_kl, width=.5, color="r", edgecolor="black", label="Ablate Random Subnetwork")

    axs[0].set_xticks(base_xs + .5, labels = x_labels, size=20)
    axs[0].tick_params(labelsize=20)
    axs[0].set_title("Language Modeling", size=30)
    axs[0].set_xlabel("Layer", size=20)
    axs[0].set_ylabel("KL Between Full and Ablated Model Logits", size=20)

    if incl_mlm:
        probe_kl = results["probe"]["mlm kl"]
        random_kl = results["probe"]["random ablate mlm kl"]

        axs[1].bar(base_xs, probe_kl, width=.5, color="blue", edgecolor="black", label="Ablate Probe Subnetwork")
        axs[1].bar(base_xs+.5, random_kl, width=.5, color="r", edgecolor="black", label="Ablate Random Subnetwork")

        axs[1].set_xticks(base_xs + .5, labels = x_labels, size=20)
        axs[1].tick_params(labelsize=20)
        axs[1].set_title("Masked Language Modeling", size=30)
        axs[1].set_xlabel("Layer", size=20)
        axs[1].set_ylabel("KL Between Full and Ablated Model Logits", size=20)

    axs[0].legend(fontsize=25)
    plt.savefig(os.path.join(outdir, "Per_Layer_LM_KL.png"))

    # Correlate layerwise KNN test accuracy with LM accuracy difference
    title = f'{model}-{task} Test Acc vs. LM Acc Difference'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    probe_diffs = results["probe"]["vanilla acc"] - results["probe"]["ablated acc"]
    probe_test_accs = results["probe"]["knn test acc"] - results["layer"]["knn test acc"]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)

    axs[0].scatter(probe_diffs, probe_test_accs)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM Accuracy Difference After Ablating Probe Subnetwork")
    axs[0].set_ylabel("Test Accuracy of Probe - Random Layer")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        probe_diffs = results["probe"]["mlm vanilla acc"] - results["probe"]["mlm ablated acc"]
        probe_test_accs = results["probe"]["knn test acc"] - results["layer"]["knn test acc"]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)
        axs[1].scatter(probe_diffs, probe_test_accs)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM Accuracy Difference After Ablating Probe Subnetwork")
        axs[1].set_ylabel("Test Accuracy of Probe - Random Layer")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Test_Acc_vs_LM_Acc.png"))

    # Correlate layerwise KNN Test Accuracy with LM KL
    title = f'{model}-{task} Test Acc vs. LM KL'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    probe_diffs = results["probe"]["kl"]
    probe_test_accs = results["probe"]["knn test acc"] - results["layer"]["knn test acc"]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)

    axs[0].scatter(probe_diffs, probe_test_accs)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM KL After Ablating Probe Subnetwork")
    axs[0].set_ylabel("Test Accuracy of Probe - Test Acc of Random Layer")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        probe_diffs = results["probe"]["mlm kl"]
        probe_test_accs = results["probe"]["knn test acc"] - results["layer"]["knn test acc"]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)
        axs[1].scatter(probe_diffs, probe_test_accs)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM KL After Ablating Probe Subnetwork")
        axs[1].set_ylabel("Test Accuracy of Probe - Test Acc of Random Layer")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Test_Acc_vs_LM_KL.png"))

    # Correlate layerwise Probe Dev Loss with LM Accuracy Difference
    title = f'{model}-{task} Test Loss vs. LM Acc Difference'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    probe_diffs = results["probe"]["vanilla acc"] - results["probe"]["ablated acc"]
    probe_loss = results["probe"]["dev loss"] - results["layer"]["dev loss"]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_loss)

    axs[0].scatter(probe_diffs, probe_loss)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM Accuracy Difference After Ablating Probe Subnetwork")
    axs[0].set_ylabel("Dev Loss of Probe")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        probe_diffs = results["probe"]["mlm vanilla acc"] - results["probe"]["mlm ablated acc"]
        probe_loss = results["probe"]["dev loss"]  - results["layer"]["dev loss"]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_loss)
        axs[1].scatter(probe_diffs, probe_loss)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM Accuracy Difference After Ablating Probe Subnetwork")
        axs[1].set_ylabel("Dev Loss of Probe - Dev Loss of Random Layer")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Dev_Loss_vs_LM_Acc.png"))

    # Correlate layerwise Probe Test Loss with LM KL
    title = f'{model}-{task} Test Loss vs. LM KL'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    probe_diffs = results["probe"]["kl"]
    probe_loss = results["probe"]["dev loss"] - results["layer"]["dev loss"]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_loss)

    axs[0].scatter(probe_diffs, probe_loss)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM KL After Ablating Probe Subnetwork")
    axs[0].set_ylabel("Dev Loss of Probe - Random Loss")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        probe_diffs = results["probe"]["mlm kl"]
        probe_loss = results["probe"]["dev loss"] - results["layer"]["dev loss"]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_loss)
        axs[1].scatter(probe_diffs, probe_loss)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM KL After Ablating Probe Subnetwork")
        axs[1].set_ylabel("Dev Loss of Probe - Random Loss")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Dev_Loss_vs_LM_KL.png"))