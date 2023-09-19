import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from Datasets import GifEvalDataset


def create_gif_dataset(config):
    dataset = GifEvalDataset(
        config["test_data_path"], config["variable"], config["probe_index"]
    )
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )


def eval_model(config, probe, dataloader, gif_vectors, key):
    # Gets vectors and labels from a probe to run KNN
    x = []
    y = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        outputs = probe(**batch)

        x.append(
            outputs["hidden_states"].detach()
        )  # These are the residual stream updates used to calculate loss
        y.append(outputs.labels)

    x = torch.cat(x).cpu().numpy()
    y = torch.cat(y).cpu().numpy()
    gif_vectors.append({key: {"x": x, "y": y}})


def create_gif(outpath, gif_vectors):
    "Constructs a gif of scatter plots of PCA updates through training"

    colors = [
        "black",
        "red",
        "lightsalmon",
        "peachpuff",
        "peru",
        "orange",
        "yellow",
        "greenyellow",
        "green",
        "aquamarine",
        "teal",
        "blue",
        "purple",
        "violet",
        "pink",
        "gray",
        "darkgoldenrod",
        "powderblue",
        "dodgerblue",
        "saddlebrown",
    ]

    img_dir = os.path.join(outpath, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(len(gif_vectors)):
        d = gif_vectors[i]
        title = list(d.keys())[0]
        vectors = d[title]["x"]
        classes = d[title]["y"]
        pca = PCA(n_components=2)
        low_dim_vectors = pca.fit_transform(vectors)
        unq_classes = np.unique(classes)
        plt.figure()
        plt.title(title)
        for idx in range(len(unq_classes)):
            cl = unq_classes[idx]
            vecs = low_dim_vectors[classes == cl]
            plt.scatter(vecs[:, 0], vecs[:, 1], c=colors[idx], label=str(cl))
        plt.savefig(os.path.join(img_dir, str(i) + ".png"))

    images = []
    for file_name in sorted(os.listdir(img_dir)):
        if file_name.endswith(".png"):
            file_path = os.path.join(img_dir, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave(
        os.path.join(outpath, "nearest_neighbors.gif"), images, duration=2000
    )
