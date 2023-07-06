import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def get_best_test_acc(df):
    idx = df["knn test acc"].idxmax()
    lm_acc_diff = df["vanilla acc"].iloc[idx].item() - df["ablated acc"].iloc[idx].item()
    lm_kl = df["kl"].iloc[idx].item()
    mlm_acc_diff = df["mlm vanilla acc"].iloc[idx].item() - df["mlm ablated acc"].iloc[idx].item()
    mlm_kl = df["mlm kl"].iloc[idx].item()
    return df["knn test acc"].iloc[idx].item(), lm_acc_diff, lm_kl, mlm_acc_diff, mlm_kl

def get_best_test_acc_diff(probe, embed):
    diffs = probe["knn test acc"] - embed["knn test acc"]
    idx = diffs.idxmax()
    lm_acc_diff = probe["vanilla acc"].iloc[idx].item() - probe["ablated acc"].iloc[idx].item()
    lm_kl = probe["kl"].iloc[idx].item()
    mlm_acc_diff = probe["mlm vanilla acc"].iloc[idx].item() - probe["mlm ablated acc"].iloc[idx].item()
    mlm_kl = probe["mlm kl"].iloc[idx].item()
    return diffs.iloc[idx].item(), lm_acc_diff, lm_kl, mlm_acc_diff, mlm_kl

def get_best_dev_loss(df):
    idx = df["dev loss"].idxmin()
    lm_acc_diff = df["vanilla acc"].iloc[idx].item() - df["ablated acc"].iloc[idx].item()
    lm_kl = df["kl"].iloc[idx].item()
    mlm_acc_diff = df["mlm vanilla acc"].iloc[idx].item() - df["mlm ablated acc"].iloc[idx].item()
    mlm_kl = df["mlm kl"].iloc[idx].item()
    return df["dev loss"].iloc[idx].item(), lm_acc_diff, lm_kl, mlm_acc_diff, mlm_kl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--task",
        default=None,
        help="task name",
        metavar="FILE",
    )
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
        raise ValueError("Must include path")
    if not hasattr(args, "task"):
        raise ValueError("Must include task")   


    bert_results= pd.read_csv(os.path.join(args.dir, "BERT", args.task, args.task.lower(), 'results.csv'))
    bert_embed = pd.read_csv(os.path.join(args.dir, "BERT", args.task, f'rand-embeds-{args.task.lower()}', 'results.csv'))
    roberta_results = pd.read_csv(os.path.join(args.dir, "RoBERTa", args.task, args.task.lower(), 'results.csv'))
    roberta_embed = pd.read_csv(os.path.join(args.dir, "RoBERTa", args.task, f'rand-embeds-{args.task.lower()}', 'results.csv'))
    mpnet_results = pd.read_csv(os.path.join(args.dir, "MPNet", args.task, args.task.lower(), 'results.csv'))
    mpnet_embed = pd.read_csv(os.path.join(args.dir, "MPNet", args.task, f'rand-embeds-{args.task.lower()}', 'results.csv'))
    electra_results = pd.read_csv(os.path.join(args.dir, "Electra", args.task, args.task.lower(), 'results.csv'))
    electra_embed = pd.read_csv(os.path.join(args.dir, "Electra", args.task, f'rand-embeds-{args.task.lower()}', 'results.csv'))

    # Set up directory
    outdir = os.path.join("../Figs", "Across_Models", args.task)
    os.makedirs(outdir, exist_ok=True)

    # Correlate best test acc with LM accuracy difference and KL
    title = f'Across Models: Best Test Acc Correlations'

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)

    
    test_acc = []
    lm_acc = []
    lm_kl = []
    mlm_acc = []
    mlm_kl = []

    outs =  get_best_test_acc(bert_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_test_acc(roberta_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_test_acc(mpnet_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_test_acc(electra_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lm_acc, test_acc)

    axs[0, 0].scatter(lm_acc, test_acc)
    axs[0, 0].set_title("Language Modeling: Acc Diff", size=10)
    axs[0, 0].set_xlabel("Accuracy Difference After Ablating Probe Subnetwork")
    axs[0, 0].set_ylabel("Best Test Accuracy of Probes Per Model")
    axs[0, 0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0, 0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0, 0].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lm_kl, test_acc)
    axs[0, 1].scatter(lm_kl, test_acc)
    axs[0, 1].set_title("Language Modeling: KL", size=10)
    axs[0, 1].set_xlabel("KL After Ablating Probe Subnetwork")
    axs[0, 1].set_ylabel("Best Test Accuracy of Probes Per Model")
    axs[0, 1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0, 1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0, 1].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mlm_acc, test_acc)
    axs[1, 0].scatter(mlm_acc, test_acc)
    axs[1, 0].set_title("Masked Language Modeling: Acc Diff", size=10)
    axs[1, 0].set_xlabel("Accuracy Difference After Ablating Probe Subnetwork")
    axs[1, 0].set_ylabel("Best Test Accuracy of Probes Per Model")
    axs[1, 0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[1, 0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[1, 0].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mlm_kl, test_acc)
    axs[1, 1].scatter(mlm_kl, test_acc)
    axs[1, 1].set_title("Masked Language Modeling: KL", size=10)
    axs[1, 1].set_xlabel("KL After Ablating Probe Subnetwork")
    axs[1, 1].set_ylabel("Best Test Accuracy of Probes Per Model")
    axs[1, 1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[1, 1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[1, 1].legend()

    plt.savefig(os.path.join(outdir, "Test_Acc_Correlations.png"))

    # Correlations with Test acc Difference from Embedding
    title = f'Across Models: Best Test Acc Relative to Embedding Model Correlations'

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)
    
    test_acc = []
    lm_acc = []
    lm_kl = []
    mlm_acc = []
    mlm_kl = []

    outs =  get_best_test_acc_diff(bert_results, bert_embed)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_test_acc_diff(roberta_results, roberta_embed)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_test_acc_diff(mpnet_results, mpnet_embed)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_test_acc_diff(electra_results, electra_embed)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lm_acc, test_acc)

    axs[0, 0].scatter(lm_acc, test_acc)
    axs[0, 0].set_title("Language Modeling: Acc Diff", size=10)
    axs[0, 0].set_xlabel("Accuracy Difference After Ablating Probe Subnetwork")
    axs[0, 0].set_ylabel("Best Test Accuracy Difference From Embedding")
    axs[0, 0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0, 0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0, 0].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lm_kl, test_acc)
    axs[0, 1].scatter(lm_kl, test_acc)
    axs[0, 1].set_title("Language Modeling: KL", size=10)
    axs[0, 1].set_xlabel("KL After Ablating Probe Subnetwork")
    axs[0, 1].set_ylabel("Best Test Accuracy Difference From Embedding")
    axs[0, 1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0, 1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0, 1].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mlm_acc, test_acc)
    axs[1, 0].scatter(mlm_acc, test_acc)
    axs[1, 0].set_title("Masked Language Modeling: Acc Diff", size=10)
    axs[1, 0].set_xlabel("Accuracy Difference After Ablating Probe Subnetwork")
    axs[1, 0].set_ylabel("Best Test Accuracy Difference From Embedding")
    axs[1, 0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[1, 0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[1, 0].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mlm_kl, test_acc)
    axs[1, 1].scatter(mlm_kl, test_acc)
    axs[1, 1].set_title("Masked Language Modeling: KL", size=10)
    axs[1, 1].set_xlabel("KL After Ablating Probe Subnetwork")
    axs[1, 1].set_ylabel("Best Test Accuracy Difference From Embedding")
    axs[1, 1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[1, 1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[1, 1].legend()

    plt.savefig(os.path.join(outdir, "Test_Acc_vs_Embeds_Correlations.png"))

    # Correlations with Dev Loss
    title = f'Across Models: Best Dev Loss Correlations'

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)

    test_acc = []
    lm_acc = []
    lm_kl = []
    mlm_acc = []
    mlm_kl = []

    outs = get_best_dev_loss(bert_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_dev_loss(roberta_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_dev_loss(mpnet_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    outs =  get_best_dev_loss(electra_results)
    test_acc.append(outs[0])
    lm_acc.append(outs[1])
    lm_kl.append(outs[2])
    mlm_acc.append(outs[3])
    mlm_kl.append(outs[4])

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lm_acc, test_acc)

    axs[0, 0].scatter(lm_acc, test_acc)
    axs[0, 0].set_title("Language Modeling: Acc Diff", size=10)
    axs[0, 0].set_xlabel("Accuracy Difference After Ablating Probe Subnetwork")
    axs[0, 0].set_ylabel("Best Dev Loss of Probes Per Model")
    axs[0, 0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0, 0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0, 0].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lm_kl, test_acc)
    axs[0, 1].scatter(lm_kl, test_acc)
    axs[0, 1].set_title("Language Modeling: KL", size=10)
    axs[0, 1].set_xlabel("KL After Ablating Probe Subnetwork")
    axs[0, 1].set_ylabel("Best Dev Loss of Probes Per Model")
    axs[0, 1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[0, 1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[0, 1].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mlm_acc, test_acc)
    axs[1, 0].scatter(mlm_acc, test_acc)
    axs[1, 0].set_title("Masked Language Modeling: Acc Diff", size=10)
    axs[1, 0].set_xlabel("Accuracy Difference After Ablating Probe Subnetwork")
    axs[1, 0].set_ylabel("Best Dev Loss of Probes Per Model")
    axs[1, 0].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[1, 0].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[1, 0].legend()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mlm_kl, test_acc)
    axs[1, 1].scatter(mlm_kl, test_acc)
    axs[1, 1].set_title("Masked Language Modeling: KL", size=10)
    axs[1, 1].set_xlabel("KL After Ablating Probe Subnetwork")
    axs[1, 1].set_ylabel("Best Dev Loss of Probes Per Model")
    axs[1, 1].plot([], [], ' ', label=f'R2: {round(r_value, 3)}')
    axs[1, 1].plot([], [], ' ', label=f'P: {round(p_value, 3)}')
    axs[1, 1].legend()

    plt.savefig(os.path.join(outdir, "Dev_Loss_Correlations.png"))