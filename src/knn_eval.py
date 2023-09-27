import numpy as np
import torch
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier as knn


def get_majority_acc(train_ys, test_ys):
    # Baseline performance of a classifier that just picks the most common
    # label in the trainset
    mode = stats.mode(train_ys).mode
    acc = test_ys == mode
    return np.sum(acc) / len(acc)


def get_knn_data(config, probe, dataloader):
    # Gets vectors and labels from a probe to run KNN
    x = []
    y = []
    torch.manual_seed(config["data_seed"])
    np.random.seed(config["data_seed"])
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        batch = {k: v.to(config["device"]) for k, v in batch.items() if k != "ungrammatical"}

        outputs = probe(**batch)

        x.append(
            outputs["hidden_states"].detach()
        )  # These are the residual stream updates used to calculate loss
        y.append(outputs.labels)

    x = torch.cat(x).cpu().numpy()
    y = torch.cat(y).cpu().numpy()
    return x, y


def knn_evaluation(
    config, probe, trainloader, testloader, num_knn_examples=None, per_class=None
):
    # Trains a KNN classifier and predicts on a held out dev and test set
    # Can either randomly sample N KNN examples or sample N per class
    assert num_knn_examples or per_class

    # Use cosine distance because that is what we train our mask with
    knn_clf = knn(metric="cosine", n_neighbors=1)

    probe.train(False)
    knn_x, knn_y = get_knn_data(config, probe, trainloader)
    knn_test_x, knn_test_y = get_knn_data(config, probe, testloader)

    np.random.seed(config["data_seed"])

    if num_knn_examples:
        # Use num_training_examples from probe trainset to train a KNN clf
        sample = np.random.choice(
            list(range(len(knn_x))), size=num_knn_examples, replace=False
        )
        knn_train_x = knn_x[sample]
        knn_train_y = knn_y[sample]

        # Use the rest of the examples from probe trainset as a dev set
        complement_sample = list(set(list(range(len(knn_x)))) - set(sample))
        knn_dev_x = knn_x[complement_sample]
        knn_dev_y = knn_y[complement_sample]

    elif per_class:
        knn_train_x = []
        knn_train_y = []
        knn_dev_x = []
        knn_dev_y = []
        classes = np.unique(knn_y)

        # Select per_class samples for training from each class, use the rest as a dev set
        for cl in classes:
            class_mask = knn_y == cl
            class_x = knn_x[class_mask]
            class_y = knn_y[class_mask]
            sample = np.random.choice(
                list(range(len(class_x))), size=per_class, replace=False
            )
            knn_train_x.append(class_x[sample])
            knn_train_y.append(class_y[sample])

            # Use the rest of the examples from probe trainset as a dev set
            complement_sample = list(set(list(range(len(class_x)))) - set(sample))
            knn_dev_x.append(class_x[complement_sample])
            knn_dev_y.append(class_y[complement_sample])

        knn_train_x = np.concatenate(knn_train_x, axis=0)
        knn_train_y = np.concatenate(knn_train_y, axis=0)
        knn_dev_x = np.concatenate(knn_dev_x, axis=0)
        knn_dev_y = np.concatenate(knn_dev_y, axis=0)

    dev_majority = get_majority_acc(knn_train_y, knn_dev_y)
    test_majority = get_majority_acc(knn_train_y, knn_test_y)

    knn_clf.fit(knn_train_x, knn_train_y)

    preds = knn_clf.predict(knn_dev_x)
    dev_acc = np.sum(knn_dev_y == preds) / len(preds)

    preds = knn_clf.predict(knn_test_x)
    test_acc = np.sum(knn_test_y == preds) / len(preds)

    outputs = {
        "dev_acc": dev_acc,
        "dev_majority": dev_majority,
        "test_acc": test_acc,
        "test_majority": test_majority,
    }

    return outputs
