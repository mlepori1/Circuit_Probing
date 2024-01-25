import os
import shutil
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers import GPT2LMHeadModel, GPT2Config
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    loading_from_pretrained,
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import transformer_lens.utils as lens_utils

from concept_erasure import LeaceEraser

import utils
from Datasets import AmnesicProbingDataset

"""
Runs amnesic probing with state of the art LEACE method
Link: https://arxiv.org/pdf/2306.03819.pdf
"""


def create_hooked(config):
    # Converts pretrained HF GPT2 to Transformer lens hooked transformer
    hf_model = GPT2LMHeadModel.from_pretrained(config["model_path"]).to(
        config["device"]
    )
    for param in hf_model.parameters():
        param.requires_grad = False

    hf_config = GPT2Config.from_pretrained(config["model_path"])
    cfg_dict = {
        "d_model": hf_config.n_embd,
        "d_head": hf_config.n_embd // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": hf_config.n_inner,
        "n_layers": hf_config.n_layer,
        "n_ctx": hf_config.n_positions,
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
        "normalization_type": "LN",
    }

    hooked_config = HookedTransformerConfig(**cfg_dict)
    state_dict = loading_from_pretrained.convert_gpt2_weights(hf_model, hooked_config)
    hooked_model = HookedTransformer(hooked_config)
    state_dict = loading_from_pretrained.fill_missing_keys(hooked_model, state_dict)
    hooked_model.load_state_dict(state_dict)

    return hooked_model


def create_amnesic_datasets(config):
    train_dataset = AmnesicProbingDataset(
        config["train_data_path"], config["variable"], config["device"]
    )
    test_dataset = AmnesicProbingDataset(
        config["test_data_path"], config["variable"], config["device"]
    )
    if "multitask" in config and config["multitask"]:
        t2_train_dataset = AmnesicProbingDataset(
            config["t2_train_data_path"], config["variable"], config["device"]
        )
        t2_test_dataset = AmnesicProbingDataset(
            config["t2_test_data_path"], config["variable"], config["device"]
        )
        return train_dataset, test_dataset, t2_train_dataset, t2_test_dataset
    else:
        return train_dataset, test_dataset


def residual_stream_patching_hook(
    resid_state,
    hook,
    position,
    erased_embedding,
):
    # Patches in erased embedding into the model
    resid_state[:, position, :] = erased_embedding
    return resid_state


def test_hook(config, hooked_transformer, dataset):
    unpatched_logits, cache = hooked_transformer.run_with_cache(dataset.x)

    activations = cache[
        lens_utils.get_act_name(config["residual_location"], config["target_layer"])
    ][:, config["target_position"], :]

    # patch in the normal embeddings
    hook_fn = partial(
        residual_stream_patching_hook,
        position=config["target_position"],
        erased_embedding=activations,
    )
    patched_logits = hooked_transformer.run_with_hooks(
        dataset.x,
        fwd_hooks=[
            (
                lens_utils.get_act_name(
                    config["residual_location"], config["target_layer"]
                ),
                hook_fn,
            )
        ],
    )

    assert torch.all(unpatched_logits == patched_logits)

    # patch in random embeddings
    hook_fn = partial(
        residual_stream_patching_hook,
        position=config["target_position"],
        erased_embedding=torch.rand(activations.shape),
    )
    patched_logits = hooked_transformer.run_with_hooks(
        dataset.x,
        fwd_hooks=[
            (
                lens_utils.get_act_name(
                    config["residual_location"], config["target_layer"]
                ),
                hook_fn,
            )
        ],
    )

    assert torch.all((unpatched_logits != patched_logits)[:, -1])


def erased_lm_eval(config, hooked_transformer, erased_embedding, dataset):
    # Test how LM performance drops after concept erasure

    # Just to verify that patching erased embeddings is working correctly
    unpatched_logits = hooked_transformer(dataset.x)

    hook_fn = partial(
        residual_stream_patching_hook,
        position=config["target_position"],
        erased_embedding=erased_embedding,
    )
    patched_logits = hooked_transformer.run_with_hooks(
        dataset.x,
        fwd_hooks=[
            (
                lens_utils.get_act_name(
                    config["residual_location"], config["target_layer"]
                ),
                hook_fn,
            )
        ],
    )

    # Assert that patching has changed the logits
    assert (
        torch.numel(unpatched_logits[:, -1])
        / torch.sum(unpatched_logits[:, -1] != patched_logits[:, -1])
        > 0.99
    )

    prediction = torch.argmax(patched_logits[:, -1], dim=-1)
    accs = prediction == dataset.y
    return (torch.sum(accs) / len(accs)).cpu().detach().numpy()


def linear_probe(config, x, y, hidden_dim=128):
    # Train a linear probe on embeddings and return train acc
    # Use default lr and epochs from linear probing experiments
    # Note that the results might differ slightly from those given by probing.py
    # because we're doing full-batch training here and the random seed is different.
    #
    # This evaluation is purely a sanity check to make sure that LEACE is working
    # correctly on our embeddings.
    loss_fn = nn.CrossEntropyLoss()
    probe = nn.Sequential(nn.Linear(hidden_dim, config["n_classes"])).to("cuda")
    optimizer = AdamW(probe.parameters(), lr=0.1)

    probe.train()
    for _ in range(100):
        out = probe(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    probe.eval()
    logits = probe(x)
    pred = torch.argmax(logits, dim=-1)
    return torch.mean((pred == y).float()).cpu().detach().numpy()


def amnesic_probing(config, hooked_transformer, dataset):
    # Fit a LEACE eraser to a dataset of activations, erase
    # the linearly-decodable information about a variable from the activation set
    # then run the model again, patching in the erased activations in the correct spot.
    # Record accuracy to see how performance drops after erasure.
    hooked_transformer.train(False)
    _, cache = hooked_transformer.run_with_cache(dataset.x)
    activations = cache[
        lens_utils.get_act_name(config["residual_location"], config["target_layer"])
    ][:, config["target_position"], :]

    pre_acc = linear_probe(config, activations, dataset.probe_var)

    # Erase linear information!
    y_one_hot = torch.nn.functional.one_hot(dataset.probe_var)
    eraser = LeaceEraser.fit(activations, y_one_hot)
    erased = eraser(activations)

    post_acc = linear_probe(config, erased, dataset.probe_var)

    accuracy = erased_lm_eval(config, hooked_transformer, erased, dataset)
    return accuracy, pre_acc, post_acc


def amnesic_probing_multitask(config, hooked_transformer, task1_dataset, task2_dataset):
    # Function used to do amnesic probing for Experiment 2's multitask setup.
    # Fit a LEACE eraser to a dataset of activations from Task 1, erase
    # the linearly-decodable information about a variable from the activation set
    # then run the model again, patching in the erased activations in the correct spot.
    # Record accuracy to see how performance drops after erasure.
    #
    # Then, use the same eraser to erase linearly decodable information from activations
    # generated by the same model run on task 2. Patch in the erased activations and
    # record accuracy to see how performance drops.

    hooked_transformer.train(False)
    _, cache = hooked_transformer.run_with_cache(task1_dataset.x)
    activations = cache[
        lens_utils.get_act_name(config["residual_location"], config["target_layer"])
    ][:, config["target_position"], :]

    task1_pre_acc = linear_probe(config, activations, task1_dataset.probe_var)

    # Erase linear information!
    y_one_hot = torch.nn.functional.one_hot(task1_dataset.probe_var)
    eraser = LeaceEraser.fit(activations, y_one_hot)
    erased = eraser(activations)

    task1_post_acc = linear_probe(config, erased, task1_dataset.probe_var)

    task1_accuracy = erased_lm_eval(config, hooked_transformer, erased, task1_dataset)

    # Apply the learned eraser to embeddings generated from task2 dataset and compute acc
    _, cache = hooked_transformer.run_with_cache(task2_dataset.x)
    activations = cache[
        lens_utils.get_act_name(config["residual_location"], config["target_layer"])
    ][:, config["target_position"], :]
    task2_erased = eraser(activations)
    task2_accuracy = erased_lm_eval(
        config, hooked_transformer, task2_erased, task2_dataset
    )

    return task1_accuracy, task1_pre_acc, task1_post_acc, task2_accuracy


def test_leace(config):
    # Baseline test of LEACE

    # Test with small number of classes
    X, Y = make_classification(
        n_samples=1000,
        n_features=128,  # hidden state dim
        n_classes=2,
        n_clusters_per_class=1,
        n_informative=4,
        random_state=0,
    )
    X_t = torch.from_numpy(X).float().to("cuda")
    Y_t = torch.from_numpy(Y).to("cuda")

    # Linear probe
    pre_acc = linear_probe(config, X_t, Y_t)
    assert pre_acc > 0.95

    # Erase linear information!
    y_one_hot = torch.nn.functional.one_hot(Y_t)
    eraser = LeaceEraser.fit(X_t, y_one_hot)
    X_ = eraser(X_t)

    # But learns nothing after
    post_acc = linear_probe(config, X_, Y_t)
    assert post_acc < 0.6

    ## Given a high dim input, with many classes and many informative features
    X, Y = make_classification(
        n_samples=1000,
        n_features=128,  # hidden state dim
        n_classes=config["n_classes"],
        n_clusters_per_class=1,
        n_informative=64,
        random_state=0,
    )
    X_t = torch.from_numpy(X).float().to("cuda")
    Y_t = torch.from_numpy(Y).to("cuda")

    # Probe learns something before
    pre_acc = linear_probe(config, X_t, Y_t)
    assert pre_acc > 0.95

    # Erase linear information!
    y_one_hot = torch.nn.functional.one_hot(Y_t)
    eraser = LeaceEraser.fit(X_t, y_one_hot)
    X_ = eraser(X_t)

    # Learns nothing after
    post_acc = linear_probe(config, X_, Y_t)
    assert post_acc < 0.05


def main():
    config = utils.get_config()
    df = pd.DataFrame()

    test_leace(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device
    shutil.rmtree(config["results_dir"], ignore_errors=True)
    os.makedirs(config["results_dir"])

    # Iterate through all hyperparameters
    for target_layer in config["target_layer_list"]:
        for target_position in config["target_position_list"]:
            for residual_location in config["residual_location_list"]:
                for model_seed in config["model_seed_list"]:
                    # Create a new model_id
                    model_id = str(uuid.uuid4())

                    config["target_layer"] = target_layer
                    config["target_position"] = target_position
                    config["residual_location"] = residual_location

                    # Force datasets to be the same
                    torch.manual_seed(config["data_seed"])
                    np.random.seed(config["data_seed"])

                    # Set Model seed
                    torch.manual_seed(model_seed)
                    np.random.seed(model_seed)
                    hooked_transformer = create_hooked(config)

                    if "multitask" in config and config["multitask"]:
                        (
                            t1_trainset,
                            t1_testset,
                            t2_trainset,
                            t2_testset,
                        ) = create_amnesic_datasets(config)
                        test_hook(config, hooked_transformer, t1_trainset)
                        (
                            train_t1_accuracy,
                            train_t1_pre_acc,
                            train_t1_post_acc,
                            train_t2_accuracy,
                        ) = amnesic_probing_multitask(
                            config, hooked_transformer, t1_trainset, t2_trainset
                        )
                        (
                            test_t1_accuracy,
                            test_t1_pre_acc,
                            test_t1_post_acc,
                            test_t2_accuracy,
                        ) = amnesic_probing_multitask(
                            config, hooked_transformer, t1_testset, t2_testset
                        )
                        output_dict = {
                            "model_id": [model_id],
                            "data_seed": [config["data_seed"]],
                            "model_seed": [model_seed],
                            "target_layer": [config["target_layer"]],
                            "residual_location": [config["residual_location"]],
                            "target_position": [config["target_position"]],
                            "model path": [config["model_path"]],
                            "Same Train Pre Erasure Probe Acc": [train_t1_pre_acc],
                            "Same Train Post Erasure Probe Acc": [train_t1_post_acc],
                            "Same Test Pre Erasure Probe Acc": [test_t1_pre_acc],
                            "Same Test Post Erasure Probe Acc": [test_t1_post_acc],
                            "Same train acc": [train_t1_accuracy],
                            "Same test acc": [test_t1_accuracy],
                            "Diff train acc": [train_t2_accuracy],
                            "Diff test acc": [test_t2_accuracy],
                        }

                    else:
                        trainset, testset = create_amnesic_datasets(config)
                        test_hook(config, hooked_transformer, trainset)

                        train_acc, pre_acc, post_acc = amnesic_probing(
                            config, hooked_transformer, trainset
                        )
                        (
                            eval_acc,
                            val_pre_acc,
                            val_post_acc,
                        ) = amnesic_probing(config, hooked_transformer, testset)

                        output_dict = {
                            "model_id": [model_id],
                            "data_seed": [config["data_seed"]],
                            "model_seed": [model_seed],
                            "target_layer": [config["target_layer"]],
                            "residual_location": [config["residual_location"]],
                            "target_position": [config["target_position"]],
                            "model path": [config["model_path"]],
                            "Train Pre Erasure Probe Acc": [pre_acc],
                            "Train Post Erasure Probe Acc": [post_acc],
                            "Test Pre Erasure Probe Acc": [val_pre_acc],
                            "Test Post Erasure Probe Acc": [val_post_acc],
                            "train acc": [train_acc],
                            "test acc": [eval_acc],
                        }

                    df = pd.concat(
                        [df, pd.DataFrame.from_dict(output_dict)],
                        ignore_index=True,
                    )

                    print("Saving csv")
                    # Will overwrite this file after every evaluation
                    df.to_csv(os.path.join(config["results_dir"], "results.csv"))


if __name__ == "__main__":
    main()
