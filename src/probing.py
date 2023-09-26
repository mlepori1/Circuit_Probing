import os
import shutil
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import GPT2LMHeadModel, GPT2Config
from transformer_lens import HookedTransformer, HookedTransformerConfig, loading_from_pretrained
import transformer_lens.utils as lens_utils

import utils
from Datasets import ProbingDataset, CounterfactualEmbeddingsDataset


def create_probe(config):
    # Probe takes input from TransformerLens Hooked Transformer, maps to output space using a small nn.module
    hf_model = GPT2LMHeadModel.from_pretrained(config["model_path"]).to(config["device"])
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

    if config["intermediate_size"] != -1:
        probe = nn.Sequential(
                nn.Linear(hf_config.n_embd, config["intermediate_size"]),
                nn.ReLU(),
                nn.Linear(config["intermediate_size"], config["n_classes"]),
            )
    else:
        probe = nn.Sequential(nn.Linear(hf_config.n_embd, config["n_classes"]))
    
    return hooked_model, probe.to(config["device"])

def create_probe_datasets(config):
    train_dataset = ProbingDataset(
        config["train_data_path"], config["variable"], config["device"]
    )
    test_dataset = ProbingDataset(
        config["test_data_path"], config["variable"], config["device"]
    )

    trainloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    testloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )
    return trainloader, testloader

def eval_probe(config, hooked_transformer, probe, dataloader):
    # Implements a simple probe evaluation loop
    loss_fn = nn.CrossEntropyLoss()

    probe.train(False)
    average_eval_loss = []
    correct = 0.0
    total = 0.0
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        _, cache = hooked_transformer.run_with_cache(batch["input_ids"])
        probed_embed = cache[lens_utils.get_act_name(config["residual_location"], config["target_layer"])][:, config["target_position"], :]
        out = probe(probed_embed)

        loss = loss_fn(out, batch["labels"])
        average_eval_loss.append(loss.detach().item())
        correct += torch.sum(torch.argmax(out.cpu(), dim=-1) == batch["labels"].cpu())
        total += len(batch["labels"])
    probe.train(True)
    loss = np.sum(average_eval_loss) / len(average_eval_loss)
    acc = correct / total
    return loss, acc.item()


def train_probe(config, hooked_transformer, probe, trainloader):
    # Implements a simple training loop that optimizes binary masks over networks
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(probe.parameters(), lr=config["lr"])
    num_training_steps = len(trainloader) * config["num_epochs"]
    progress_bar = tqdm(range(num_training_steps))

    probe.train()
    for epoch in range(config["num_epochs"]):
        progress_bar.set_description(f"Training epoch {epoch}")
        average_train_loss = []
        for batch in trainloader:
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            _, cache = hooked_transformer.run_with_cache(batch["input_ids"])
            probed_embed = cache[lens_utils.get_act_name(config["residual_location"], config["target_layer"])][:, config["target_position"], :]
            out = probe(probed_embed)

            loss = loss_fn(out, batch["labels"])
            loss.backward()
            average_train_loss.append(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        train_loss = np.sum(average_train_loss) / len(average_train_loss)

        progress_bar.set_postfix(
            {
                "Train Loss": round(train_loss, 4),
            }
        )
    return probe


def get_counterfactual_dataset(config):
    return CounterfactualEmbeddingsDataset(config["test_data_path"], config["counterfactual_label"], config["counterfactual_variable"], config["device"])

def train_counterfactual_embedding(config, counterfactual_embedding, probe, sample):
    # For each sample, update the embedding such that it produces a counterfactual answer in the probe
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW([counterfactual_embedding], lr=config["lr"])
    pre_state_dict = probe.state_dict()
    probe.train(False)
    for _ in range(config["num_epochs"]):
        train_loss = []
        out = probe(counterfactual_embedding.reshape(1, -1))[0]
        loss = loss_fn(out, sample["counterfactual_variables"])
        loss.backward()
        train_loss.append(loss.detach().item())
        optimizer.step()
        optimizer.zero_grad()

    # assert that probe did not change during embedding training
    post_state_dict = probe.state_dict()
    for k in pre_state_dict.keys():
        assert torch.all(pre_state_dict[k] == post_state_dict[k])
    out = probe(counterfactual_embedding)
    success = torch.argmax(out, dim=-1)[0] == sample["counterfactual_variables"]
    return counterfactual_embedding, success

def residual_stream_patching_hook(
    resid_state,
    hook,
    position,
    counterfactual_embedding,
):
    # Patches in counterfactual embedding into the model
    resid_state[:, position, :] = counterfactual_embedding
    return resid_state

def compute_logit_difference(
        patched_logits,
        untouched_logits,
        sample
):
    patched_diff = (patched_logits[0, -1, sample["counterfactual_labels"]] - patched_logits[0, -1, sample["original_labels"]]).cpu().item()
    untouched_diff = (untouched_logits[0, -1, sample["counterfactual_labels"]] - untouched_logits[0, -1, sample["original_labels"]]).cpu().item()
    return patched_diff/untouched_diff

def counterfactual_lm_eval(config, hooked_transformer, counterfactual_embedding, sample):
    # If the counterfactual embedding works, then it should produce counterfactual behavior in the overall model
    # Test whether this happens
    hook_fn = partial(residual_stream_patching_hook, position=config["target_position"], counterfactual_embedding=counterfactual_embedding)
    patched_logits = hooked_transformer.run_with_hooks(sample["input_ids"], fwd_hooks=[
        (lens_utils.get_act_name(config["residual_location"], config["target_layer"]), hook_fn)
    ])
    untouched_logits = hooked_transformer(sample["input_ids"])
    logit_diff = compute_logit_difference(patched_logits, untouched_logits, sample)
    prediction = torch.argmax(patched_logits[0, -1])
    predicts_counterfactual = prediction == sample["counterfactual_labels"]
    predicts_original = prediction == sample["original_labels"]
    return predicts_counterfactual, predicts_original, logit_diff

def patching_test(config, hooked_transformer, original_embedding, sample):
    # If the counterfactual embedding works, then it should produce counterfactual behavior in the overall model
    # Test whether this happens
    hook_fn = partial(residual_stream_patching_hook, position=config["target_position"], counterfactual_embedding=original_embedding)
    patched_logits = hooked_transformer.run_with_hooks(sample["input_ids"], fwd_hooks=[
        (lens_utils.get_act_name(config["residual_location"], config["target_layer"]), hook_fn)
    ])
    untouched_logits = hooked_transformer(sample["input_ids"])
    assert torch.all(patched_logits == untouched_logits)
    prediction = torch.argmax(patched_logits[0, -1])
    assert prediction == sample["original_labels"]

def counterfactual_embedding_eval(config, hooked_transformer, probe, dataset):
    # Counterfactual embedding evaluation trains a counterfactual embedding for all samples,
    # verifies that this embedding changes probe behavior, and then analyzes its impact on downstream model behavior
    # Train embedding to make probe predict counterfactual_variable. Investigate whether full model (with CF embed patched)
    # predicts the counterfactual_label, also record when it still predicts the original_label, and also record the normalized logit difference
    hooked_transformer.train(False)
    probe.train(False)
    for parameter in probe.parameters():
        parameter.requires_grad = False
    counterfactual_embed_success = []
    counterfactual_lm_cf_label = []
    counterfactual_lm_original_label = []
    logit_diffs = []
    total_examples = 0
    
    progress_bar = tqdm(range(len(dataset)))
    for idx, sample in enumerate(dataset):
        # Run this evaluation if the counterfactual produces a different label
        if sample["original_labels"] != sample["counterfactual_labels"]:                
            _, cache = hooked_transformer.run_with_cache(sample["input_ids"])
            counterfactual_embedding = nn.Parameter(cache[lens_utils.get_act_name(config["residual_location"], config["target_layer"])][:, config["target_position"], :])
            # Verify that patching is working as intended
            if idx == 0:
                patching_test(config, hooked_transformer, counterfactual_embedding, sample)
            counterfactual_embedding, success = train_counterfactual_embedding(config, counterfactual_embedding, probe, sample)
            counterfactual_embed_success.append(success.cpu())
            predicts_counterfactual, predicts_original, logit_diff = counterfactual_lm_eval(config, hooked_transformer, counterfactual_embedding, sample)
            counterfactual_lm_cf_label.append(predicts_counterfactual.cpu())
            counterfactual_lm_original_label.append(predicts_original.cpu())
            logit_diffs.append(logit_diff)
            total_examples += 1
        progress_bar.update(1)

    return np.sum(counterfactual_embed_success)/len(counterfactual_embed_success), np.sum(counterfactual_lm_cf_label)/len(counterfactual_lm_cf_label), np.sum(counterfactual_lm_original_label)/len(counterfactual_lm_original_label), np.mean(logit_diffs), total_examples

def get_dataset_length(dataloader):
    length = 0
    for batch in dataloader:
        length += len(batch["input_ids"])
    if length == 0:
        return length
    return length


def main():
    config = utils.get_config()
    df = pd.DataFrame()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device
    shutil.rmtree(config["results_dir"], ignore_errors=True)
    os.makedirs(config["results_dir"])

    # Iterate through all training hyperparameters
    for lr in config["lr_list"]:
        for batch_size in config["batch_size_list"]:
            for target_layer in config["target_layer_list"]:
                for target_position in config["target_position_list"]:
                    for residual_location in config["residual_location_list"]:
                        for model_seed in config["model_seed_list"]:
                            # Create a new model_id
                            model_id = str(uuid.uuid4())

                            config["lr"] = lr
                            config["batch_size"] = batch_size

                            config["target_layer"] = target_layer
                            config["target_position"] = target_position
                            config["residual_location"] = residual_location

                            # Force datasets to be the same
                            torch.manual_seed(config["data_seed"])
                            np.random.seed(config["data_seed"])

                            trainloader, testloader = create_probe_datasets(config)

                            # Get number of labelled tokens per dataset
                            train_length = get_dataset_length(trainloader)
                            test_length = get_dataset_length(testloader)


                            # Set Model seed
                            torch.manual_seed(model_seed)
                            np.random.seed(model_seed)
                            hooked_transformer, probe = create_probe(config)

                            if config["num_epochs"] != 0:
                                probe = train_probe(
                                    config,
                                    hooked_transformer,
                                    probe,
                                    trainloader,
                                )
                            final_train_loss, final_train_acc = eval_probe(
                                config, hooked_transformer, probe, trainloader
                            )
                            final_eval_loss, final_eval_acc = eval_probe(
                                config, hooked_transformer, probe, testloader
                            )

                            output_dict = {
                                "model_id": [model_id],
                                "train length": [train_length],
                                "test length": [test_length],
                                "batch_size": [config["batch_size"]],
                                "lr": [config["lr"]],
                                "data_seed": [config["data_seed"]],
                                "model_seed": [model_seed],
                                "target_layer": [config["target_layer"]],
                                "residual_location": [config["residual_location"]],
                                "target_position": [config["target_position"]],
                                "model path": [config["model_path"]],
                                "train loss": [final_train_loss],
                                "test loss": [final_eval_loss],
                                "train acc": [final_train_acc],
                                "test acc": [final_eval_acc],
                            }

                            if config["counterfactual_embeddings"]:
                                testset = get_counterfactual_dataset(config)
                                embed_success, predicts_cf, predicts_original, logit_diffs, total_examples = counterfactual_embedding_eval(config, hooked_transformer, probe, testset)
                                output_dict["counterfactual embedding success"] = [embed_success]
                                output_dict["counterfactual predicts CF label"] = [predicts_cf]
                                output_dict["counterfactual predicts original label"] = [predicts_original]
                                output_dict["counterfactual average logit diff"] = [logit_diffs]
                                output_dict["counterfactual dataset size"] = [total_examples]

                            df = pd.concat(
                                [df, pd.DataFrame.from_dict(output_dict)],
                                ignore_index=True,
                            )

                            print("Saving csv")
                            # Will overwrite this file after every evaluation
                            df.to_csv(
                                os.path.join(config["results_dir"], "results.csv")
                            )


if __name__ == "__main__":
    main()
