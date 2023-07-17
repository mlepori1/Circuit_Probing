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
    #probe_results = probe_results[probe_results["operation"] == "attn"]
    layer_results = pd.read_csv(os.path.join(args.dir, f'rand-layer-{suffix}', 'results.csv'))
    #layer_results = layer_results[layer_results["operation"] == "attn"]

    results = {
        "probe": probe_results,
        "layer": layer_results
    }

    # Set up directory
    path, task = os.path.split(args.dir)
    _, model = os.path.split(path)
    outdir = os.path.join("../Figs", model, task)
    os.makedirs(outdir, exist_ok=True)

    # Correlate layerwise Dev Loss with LM KL
    title = f'{model}-{task} Dev Loss vs. LM KL'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    probe_diffs = results["probe"]["kl"] #/ results["probe"]["random ablate kl"]
    probe_test_accs = results["probe"]["dev loss"] #/ results["layer"]["dev loss"]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)

    axs[0].scatter(probe_diffs, probe_test_accs)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM KL After Ablating Probe Subnetwork")
    axs[0].set_ylabel("Test Accuracy of Probe - Test Acc of Random Layer")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        probe_diffs = results["probe"]["mlm kl"] #/ results["probe"]["random ablate mlm kl"]
        probe_test_accs = results["probe"]["dev loss"] #/ results["layer"]["dev loss"]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)
        axs[1].scatter(probe_diffs, probe_test_accs)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM KL After Ablating Probe Subnetwork")
        axs[1].set_ylabel("Test Accuracy of Probe - Test Acc of Random Layer")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Dev_Loss_vs_LM_KL.png"))

    # Correlate layerwise KNN Test Acc with LM KL
    title = f'{model}-{task} Test Acc vs. LM KL'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    results
    probe_diffs = results["probe"]["kl"] #/ results["probe"]["random ablate kl"]
    probe_test_accs = results["probe"]["knn test acc"] #/ results["layer"]["knn test acc"]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)

    axs[0].scatter(probe_diffs, probe_test_accs)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM KL After Ablating Probe Subnetwork")
    axs[0].set_ylabel("Test Accuracy of Probe - Test Acc of Random Layer")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        probe_diffs = results["probe"]["mlm kl"] #/ results["probe"]["random ablate mlm kl"]
        probe_test_accs = results["probe"]["knn test acc"] #/ results["layer"]["knn test acc"]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(probe_diffs, probe_test_accs)
        axs[1].scatter(probe_diffs, probe_test_accs)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM KL After Ablating Probe Subnetwork")
        axs[1].set_ylabel("Test Accuracy of Probe - Test Acc of Random Layer")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Test_Acc_vs_LM_KL.png"))
