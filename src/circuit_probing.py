import os
import shutil
import uuid

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

import gif_eval
import utils
from knn_eval import knn_evaluation
from lm_eval import agreement_eval, reflexive_eval, get_lm_eval_data, lm_eval, agreement_qualitative_eval


class TemperatureCallback:
    # A simple callback that updates the probes temperature parameter,
    # which transforms a soft mask into a hard mask
    def __init__(self, total_epochs, final_temp):
        self.temp_increase = final_temp ** (1.0 / total_epochs)

    def update(self, model):
        temp = model.temperature
        model.temperature = temp * self.temp_increase


def eval_probe(config, probe, dataloader):
    # Implements a simple probe evaluation loop
    probe.train(False)
    average_eval_loss = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        batch = {k: v.to(config["device"]) for k, v in batch.items() if k != "ungrammatical"}
        out = probe(**batch)
        average_eval_loss.append(out.loss.detach().item())
    probe.train(True)
    return np.sum(average_eval_loss) / len(average_eval_loss)


def train_probe(config, probe, trainloader, gif_dataset=None, gif_vectors=None):
    # Implements a simple training loop that optimizes binary masks over networks
    temp_callback = TemperatureCallback(config["num_epochs"], config["max_temp"])
    optimizer = AdamW(probe.parameters(), lr=config["lr"])
    num_training_steps = len(trainloader) * config["num_epochs"]
    progress_bar = tqdm(range(num_training_steps))

    probe.train()
    for epoch in range(config["num_epochs"]):
        progress_bar.set_description(f"Training epoch {epoch}")
        average_train_loss = []
        for batch in trainloader:
            batch = {k: v.to(config["device"]) for k, v in batch.items() if k != "ungrammatical"}
            out = probe(**batch)
            loss = out.loss
            loss.backward()
            average_train_loss.append(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        train_loss = np.sum(average_train_loss) / len(average_train_loss)
        l0_statistics = probe.wrapped_model.wrapped_model.compute_l0_statistics()

        progress_bar.set_postfix(
            {
                "Train Loss": round(train_loss, 4),
                "L0 Norm": l0_statistics["total_l0"].detach().item(),
                "L0 Max": l0_statistics["max_l0"],
            }
        )

        if config["gif"]:
            if epoch % config["gif_interval"] == 0:
                gif_eval.eval_model(config, probe, gif_dataset, gif_vectors, str(epoch))

        probe.train(True)
        temp_callback.update(probe.wrapped_model.wrapped_model)

    return probe


def get_dataset_length(dataloader):
    # Get the number of labeled tokens in the dataset
    length = 0
    for batch in dataloader:
        length += torch.sum(batch["token_mask"])
    if length == 0:
        return length
    return length.item()


def main():
    config = utils.get_config()
    df = pd.DataFrame()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device
    shutil.rmtree(config["results_dir"], ignore_errors=True)
    os.makedirs(config["results_dir"])

    if config["save_models"]:
        shutil.rmtree(config["model_dir"], ignore_errors=True)
        os.makedirs(os.path.join(config["model_dir"]), exist_ok=True)

    # Iterate through all training hyperparameters
    for lr in config["lr_list"]:
        for batch_size in config["batch_size_list"]:
            for target_layer in config["target_layer_list"]:
                for probe_index in config["probe_index_list"]:
                    for operation in config["operation_list"]:
                        for mask_init in config["mask_init_list"]:
                            for model_seed in config["model_seed_list"]:
                                # Create a new model_id
                                model_id = str(uuid.uuid4())

                                config["lr"] = lr
                                config["batch_size"] = batch_size

                                config["target_layer"] = target_layer
                                config["probe_index"] = probe_index
                                config["operation"] = operation
                                config["mask_init_value"] = mask_init

                                # Set Model seed
                                torch.manual_seed(model_seed)
                                np.random.seed(model_seed)
                                model = utils.get_model(config)
                                if config["task"] == "agreement" or config["task"] == "reflexive" or config["task"] == "syntactic_number":
                                    tokenizer = GPT2Tokenizer.from_pretrained(
                                        config["model_path"]
                                    )

                                # Force datasets to be the same
                                torch.manual_seed(config["data_seed"])
                                np.random.seed(config["data_seed"])

                                if config["task"] == "agreement":
                                    (
                                        trainloader,
                                        testloader,
                                        genloader,
                                    ) = utils.create_sv_datasets(config, tokenizer)
                                elif config["task"] == "reflexive":
                                    (
                                        trainloader,
                                        testloader,
                                        male_genloader,
                                        female_testloader,
                                        female_genloader
                                    ) = utils.create_reflexive_datasets(config, tokenizer)
                                elif config["task"] == "syntactic_number":
                                    (
                                        trainloader,
                                        testloader,
                                        lm_testloader,
                                    ) = utils.create_syntactic_number_datasets(config, tokenizer)
                                else:
                                    trainloader, testloader = utils.create_datasets(
                                        config
                                    )

                                # Get number of labelled tokens per dataset
                                train_length = get_dataset_length(trainloader)
                                test_length = get_dataset_length(testloader)

                                probe = utils.create_circuit_probe(config, model)
                                probe.to(device)

                                if config["gif"]:
                                    gif_vectors = []
                                    gif_dataset = gif_eval.create_gif_dataset(config)
                                    probe.wrapped_model.wrapped_model.use_masks(False)
                                    gif_eval.eval_model(
                                        config, probe, gif_dataset, gif_vectors, "Full"
                                    )
                                    probe.wrapped_model.wrapped_model.use_masks(True)
                                else:
                                    gif_vectors = None
                                    gif_dataset = None

                                # Get full model KNN results
                                probe.wrapped_model.wrapped_model.use_masks(False)
                                pre_knn_results = knn_evaluation(
                                    config, probe, trainloader, testloader, per_class=1
                                )
                                probe.wrapped_model.wrapped_model.use_masks(True)

                                if config["num_epochs"] != 0:
                                    probe = train_probe(
                                        config,
                                        probe,
                                        trainloader,
                                        gif_dataset,
                                        gif_vectors,
                                    )
                                final_train_loss = eval_probe(
                                    config, probe, trainloader
                                )
                                final_eval_loss = eval_probe(config, probe, testloader)

                                # Run KNN subnetwork evaluation
                                knn_results = knn_evaluation(
                                    config, probe, trainloader, testloader, per_class=1
                                )

                                l0_statistics = (
                                    probe.wrapped_model.wrapped_model.compute_l0_statistics()
                                )

                                output_dict = {
                                    "model_id": [model_id],
                                    "task": [config["task"]],
                                    "train length": [train_length],
                                    "test length": [test_length],
                                    "batch_size": [config["batch_size"]],
                                    "lr": [config["lr"]],
                                    "data_seed": [config["data_seed"]],
                                    "model_seed": [model_seed],
                                    "target_layer": [config["target_layer"]],
                                    "operation": [config["operation"]],
                                    "probe index": [config["probe_index"]],
                                    "mask_init": [mask_init],
                                    "model path": [config["model_path"]],
                                    "train loss": [final_train_loss],
                                    "test loss": [final_eval_loss],
                                    "knn dev acc": [knn_results["dev_acc"]],
                                    "dev majority acc": [knn_results["dev_majority"]],
                                    "knn test acc": [knn_results["test_acc"]],
                                    "test majority acc": [knn_results["test_majority"]],
                                    "full model knn dev acc": [
                                        pre_knn_results["dev_acc"]
                                    ],
                                    "full model knn test acc": [
                                        pre_knn_results["test_acc"]
                                    ],
                                    "L0 Norm": [l0_statistics["total_l0"].cpu().item()],
                                    "L0 Max": [l0_statistics["max_l0"]],
                                }

                                if config["lm_eval"] == True:
                                    # Run language modeling evaluation to see the effect of ablating subnetworks
                                    # Extract underlying LM Model
                                    probe.train(False)
                                    model = probe.wrapped_model.wrapped_model

                                    lm_loaders = []
                                    if config["task"] == "multitask":
                                        lm_loaders.append(
                                            get_lm_eval_data(
                                                config, config["lm_same_task_path"]
                                            )
                                        )
                                        lm_loaders.append(
                                            get_lm_eval_data(
                                                config, config["lm_different_task_path"]
                                            )
                                        )
                                        lm_loader_labels = ["Same", "Different"]
                                        ablate_sets = [None]
                                        # Get ablate subnetworks in particular weight tensors
                                        ablate_sets += [
                                            "wrapped_model." + layer
                                            for layer in model.config.target_layers
                                        ]
                                    elif config["task"] == "agreement":
                                        lm_loaders = [testloader, genloader]
                                        lm_loader_labels = ["IID", "Gen"]
                                        sing_id = tokenizer(" is")["input_ids"][0] # Prepend space bc of GPT2 Tokenizer
                                        plur_id = tokenizer(" are")["input_ids"][0]
                                        ablate_sets = [None]
                                    elif config["task"] == "reflexive":
                                        lm_loaders = [testloader, male_genloader, female_testloader, female_genloader]
                                        lm_loader_labels = ["Male IID", "Male Gen", "Female IID", "Female Gen"]
                                        ablate_sets = [None]       
                                    elif config["task"] == "syntactic_number":
                                        lm_loaders = [lm_testloader]
                                        lm_loader_labels = ["IID"]
                                        sing_id = tokenizer(" is")["input_ids"][0] # Prepend space bc of GPT2 Tokenizer
                                        plur_id = tokenizer(" are")["input_ids"][0]
                                        ablate_sets = [None]
                                    else:
                                        lm_loader_labels = ["Test"]
                                        lm_loaders.append(
                                            get_lm_eval_data(
                                                config, config["test_data_path"]
                                            )
                                        )
                                        ablate_sets = [None]

                                    # Ablate subnetworks and run lm evaluation
                                    # Ablate sets are sets of masks to ablate. 
                                    # **Note on counterintuitive naming** None means that all available masks will be ablated, 
                                    # i.e. you are not passing in an ablate set
                                    for ablate_set in ablate_sets:
                                        for idx, lm_loader in enumerate(lm_loaders):
                                            model.set_ablate_mode("zero_ablate")

                                            if config["task"] == "agreement" or config["task"] == "syntactic_number":
                                                lm_results = agreement_eval(
                                                    config,
                                                    model,
                                                    lm_loader,
                                                    sing_id,
                                                    plur_id,
                                                    ablate_set,
                                                )
                                                if lm_loader_labels[idx] == "IID":
                                                    os.makedirs(os.path.join(config["results_dir"], "Qualitative"), exist_ok=True)
                                                    agreement_qualitative_eval(config, os.path.join(config["results_dir"], "Qualitative", f"{str(target_layer)}_{operation}.txt"), model, tokenizer, lm_loader, ablate_set)
                                            
                                            elif config["task"] == "reflexive":
                                                lm_results = reflexive_eval(
                                                    config,
                                                    model,
                                                    lm_loader,
                                                    ablate_set,
                                                )
                                            else:
                                                lm_results = lm_eval(
                                                    config, model, lm_loader, ablate_set
                                                )

                                            if ablate_set is None:
                                                prefix = ""
                                            else:
                                                prefix = ablate_set.split(".")[-1]

                                            output_dict[
                                                f"{prefix} vanilla acc {lm_loader_labels[idx]}"
                                            ] = [lm_results["vanilla_acc"]]
                                            output_dict[
                                                f"{prefix} ablated acc {lm_loader_labels[idx]}"
                                            ] = [lm_results["ablated_acc"]]

                                            # Ablate random subnetworks and rerun LM evaluation
                                            if (
                                                config["num_epochs"] != 0
                                                and config["num_random_ablations"] != 0
                                            ):
                                                # Can configure N random samples/reruns
                                                random_ablated_accs = []
                                                for _ in range(
                                                    config["num_random_ablations"]
                                                ):
                                                    model.set_ablate_mode(
                                                        "complement_sampled",
                                                        force_resample=True,
                                                    )

                                                    try:
                                                        # Try to run complement_sampled ablation
                                                        # If discovered mask doesn't allow for this,
                                                        # consider it a failure and return -1
                                                        # Complement sampled ablations are samples 
                                                        # from the complement of the discovered mask
                                                        if config["task"] == "agreement" or config["task"] == "syntactic_number":
                                                            random_ablate_lm_results = (
                                                                agreement_eval(
                                                                    config,
                                                                    model,
                                                                    lm_loader,
                                                                    sing_id,
                                                                    plur_id,
                                                                    ablate_set,
                                                                )
                                                            )
                                                        elif config["task"] == "reflexive":
                                                            random_ablate_lm_results = (
                                                                reflexive_eval(
                                                                    config,
                                                                    model,
                                                                    lm_loader,
                                                                    ablate_set,
                                                                )
                                                            )
                                                        else:
                                                            random_ablate_lm_results = (
                                                                lm_eval(
                                                                    config,
                                                                    model,
                                                                    lm_loader,
                                                                    ablate_set,
                                                                )
                                                            )
                                                    except:
                                                        random_ablate_lm_results = {
                                                            "ablated_acc": [-1]
                                                        }

                                                    random_ablated_accs.append(
                                                        random_ablate_lm_results[
                                                            "ablated_acc"
                                                        ]
                                                    )

                                                output_dict[
                                                    f"random ablate acc mean {lm_loader_labels[idx]}"
                                                ] = [np.mean(random_ablated_accs)]
                                                output_dict[
                                                    f"random ablate acc std {lm_loader_labels[idx]}"
                                                ] = [np.std(random_ablated_accs)]

                                        model.set_ablate_mode("none")

                                df = pd.concat(
                                    [df, pd.DataFrame.from_dict(output_dict)],
                                    ignore_index=True,
                                )

                                print("Saving csv")
                                # Will overwrite this file after every evaluation
                                df.to_csv(
                                    os.path.join(config["results_dir"], "results.csv")
                                )
                                if config["gif"]:
                                    gif_eval.create_gif(
                                        os.path.join(
                                            config["results_dir"], str(model_id)
                                        ),
                                        gif_vectors,
                                    )

                                if config["save_models"]:
                                    torch.save(
                                        probe.state_dict(),
                                        os.path.join(
                                            config["model_dir"], model_id + ".pt"
                                        ),
                                    )


if __name__ == "__main__":
    main()
