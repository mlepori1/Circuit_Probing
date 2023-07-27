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
        agg_layer_results["acc avg"].append(grp["dev acc"].mean())
        agg_layer_results["acc std"].append(grp["dev acc"].std())
        agg_layer_results["loss avg"].append(grp["dev loss"].mean())
        agg_layer_results["loss std"].append(grp["dev loss"].std())
    agg_layer_results = pd.DataFrame.from_dict(agg_layer_results)

    results = {
        "probe": probe_results,
        "layer": agg_layer_results
    }

    for i in range(len(results["probe"]["dev loss"])):
        loss_z = zscore(results["probe"]["dev loss"][i], results["layer"]["loss avg"][i], results["layer"]["loss std"][i])
        print(loss_z)
    for i in range(len(results["probe"]["dev acc"])):
        acc_z = zscore(results["probe"]["dev acc"][i], results["layer"]["acc avg"][i], results["layer"]["acc std"][i])
        #print(acc_z)