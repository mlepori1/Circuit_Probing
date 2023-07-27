import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def zscore(x, mean, std): 
    if std != 0: 
        return (x - mean)/(std)
    else:
        return None

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
    layer_results = pd.read_csv(os.path.join(args.dir, f'rand-layer-{suffix}', 'results.csv'))

    # Set up directory
    path, task = os.path.split(args.dir)
    _, model = os.path.split(path)
    outdir = os.path.join("../Figs", model, task)
    os.makedirs(outdir, exist_ok=True)

    # aggregate random layer results
    agg_layer_results = {
        "layer": [],
        "operation": [],
        "acc avg": [],
        "acc std": [],
        "loss avg": [],
        "loss std": []
    }
    grouped = layer_results.groupby(["target_layer", "operation"])
    for name, grp in grouped:
        agg_layer_results["layer"].append(name[0])
        agg_layer_results["operation"].append(name[1])
        agg_layer_results["acc avg"].append(grp["knn test acc"].mean())
        agg_layer_results["acc std"].append(grp["knn test acc"].std())
        agg_layer_results["loss avg"].append(grp["dev loss"].mean())
        agg_layer_results["loss std"].append(grp["dev loss"].std())
    agg_layer_results = pd.DataFrame.from_dict(agg_layer_results)

    results = {
        "probe": probe_results,
        "layer": agg_layer_results
    }
    # Correlate layerwise Dev Loss with LM KL
    title = f'{model}-{task} Dev Loss vs. LM KL'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    all_kls = []
    kls = []
    losses = []
    for i in range(len(results["probe"]["kl"])):
        kl_z = zscore(results["probe"]["kl"][i], results["probe"]["random ablate kl mean"][i], results["probe"]["random ablate kl std"][i])
        loss_z = zscore(results["probe"]["dev loss"][i], results["layer"]["loss avg"][i], results["layer"]["loss std"][i])
        all_kls.append(kl_z)
        if kl_z is not None and loss_z is not None and loss_z < -1:
            kls.append(kl_z)
            losses.append(loss_z)
        print(loss_z)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(kls, losses)

    axs[0].scatter(kls, losses)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM KL Z-score")
    axs[0].set_ylabel("Loss Z-Score")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        kls = []
        losses = []
        for i in range(len(results["probe"]["mlm kl"])):
            kl_z = zscore(results["probe"]["mlm kl"][i], results["probe"]["random ablate mlm kl mean"][i], results["probe"]["random ablate mlm kl std"][i])
            loss_z = zscore(results["probe"]["dev loss"][i], results["layer"]["loss avg"][i], results["layer"]["loss std"][i])
            if kl_z is not None and loss_z is not None and loss_z < -1:
                kls.append(kl_z)
                losses.append(loss_z)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(kls, losses)
        axs[1].scatter(kls, losses)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM KL Zscore")
        axs[1].set_ylabel("Loss Zscore")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Dev_Loss_vs_LM_KL.png"))

    plt.figure()
    plt.bar(range(len(all_kls)), all_kls)
    plt.ylim(-1, 10)
    plt.title("Circuit Probe Standardized KL Divergence Per Layer")
    plt.xlabel("LM Operations")
    plt.xticks(range(len(all_kls)))
    plt.ylabel("KL Divergence Z-Score")
    plt.savefig(os.path.join(outdir, "KL_Zscores.png"))

    # Correlate layerwise KNN Test Acc with LM KL
    title = f'{model}-{task} Test Acc vs. LM KL'
    if results["probe"]["mlm kl"].iloc[0].item() != -1:
        incl_mlm = True
        fig, axs = plt.subplots(2, figsize=(5, 10))
    else:
        incl_mlm = False
        fig, axs = plt.subplots(2, figsize=(5, 10))

    fig.suptitle(title, fontsize=14)

    kls = []
    accs = []
    for i in range(len(results["probe"]["kl"])):
        kl_z = zscore(results["probe"]["kl"][i], results["probe"]["random ablate kl mean"][i], results["probe"]["random ablate kl std"][i])
        acc_z = zscore(results["probe"]["knn test acc"][i], results["layer"]["acc avg"][i], results["layer"]["acc std"][i])
        if kl_z is not None and acc_z is not None and acc_z > 1:
            kls.append(kl_z)
            accs.append(acc_z)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(kls, accs)

    axs[0].scatter(kls, accs)
    axs[0].set_title("Language Modeling", size=10)
    axs[0].set_xlabel("LM KL Zscore")
    axs[0].set_ylabel("Acc Zscore")
    axs[0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0].legend()

    if incl_mlm:
        kls = []
        accs = []
        for i in range(len(results["probe"]["mlm kl"])):
            kl_z = zscore(results["probe"]["mlm kl"][i], results["probe"]["random ablate mlm kl mean"][i], results["probe"]["random ablate mlm kl std"][i])
            acc_z = zscore(results["probe"]["knn test acc"][i], results["layer"]["acc avg"][i], results["layer"]["acc std"][i])
            if kl_z is not None and acc_z is not None and acc_z > 1:
                kls.append(kl_z)
                accs.append(acc_z)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(kls, accs)
        axs[1].scatter(kls, accs)
        axs[1].set_title("Masked Language Modeling", size=10)
        axs[1].set_xlabel("MLM KL Zscore")
        axs[1].set_ylabel("Acc Zscore")
        axs[1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
        axs[1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
        axs[1].legend()

    plt.savefig(os.path.join(outdir, "Test_Acc_vs_LM_KL.png"))
