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

    # Set up directory
    path, task = os.path.split(args.dir)
    path, model = os.path.split(path)
    path, _ =os.path.split(path)

    outdir = os.path.join("../Figs", model, task)
    os.makedirs(outdir, exist_ok=True)

    baseline = pd.read_csv(os.path.join(path, "Baseline", model, task, suffix, 'results.csv'))

    # Per layer, plot probe Dev and Test accuracy, and plot performance of every other configuration for comparison
    title = f'{model}-{task} Circuit Probe Vs Linear Acc'
    fig = plt.figure(figsize=(30, 10))
    n_layers = len(probe_results["model_id"])
    base_xs = np.array((range(0, int(n_layers * 2), 2)))
    x_labels = list(range(n_layers))
    
    plt.bar(base_xs, probe_results["knn dev acc"], width=.25, color="blue", edgecolor="black", label="Probe-Train")
    plt.bar(base_xs + .25, probe_results["knn test acc"], width=.25, color="blue", hatch="O", edgecolor="black", label="Probe-Test")
    plt.bar(base_xs + .5, baseline["train acc"], width=.25, color="red", hatch="", edgecolor="black", label="Baseline-Train")
    plt.bar(base_xs + .75, baseline["dev acc"], width=.25, color="red", hatch="O", edgecolor="black", label="Baseline-Test")


    plt.xticks(base_xs + .5, labels = x_labels)
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.xlabel("Operation")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(outdir, "KNN_vs_linear.png"))
