import utils
import uuid
import os
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from knn_eval import knn_evaluation
from lm_eval import get_lm_eval_data, lm_eval, masked_lm_eval


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
        out = probe(**batch)
        average_eval_loss.append(out.loss.detach().item())
    probe.train(True)
    return np.sum(average_eval_loss) / len(average_eval_loss)


def train_probe(config, probe, trainloader, testloader):
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
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            out = probe(**batch)
            loss = out.loss
            loss.backward()
            average_train_loss.append(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        train_loss = np.sum(average_train_loss) / len(average_train_loss)

        progress_bar.set_description("Evaluation")
        eval_loss = eval_probe(config, probe, testloader)

        l0_statistics = probe.wrapped_model.model.compute_l0_statistics()

        progress_bar.set_postfix(
            {
                "Train Loss": round(train_loss, 4),
                "Eval Loss": round(eval_loss, 4),
                "L0 Norm": l0_statistics["total_l0"].detach().item(),
                "L0 Max": l0_statistics["max_l0"],
            }
        )

        probe.train(True)
        temp_callback.update(probe.wrapped_model.model)

    return probe


def get_dataset_length(dataloader):
    # Get the number of labeled tokens in the dataset
    length = 0
    for batch in dataloader:
        length += torch.sum(batch["token_mask"])
    return length.item()


def main():
    config = utils.get_config()
    df = pd.DataFrame()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device

    # Iterate through all training hyperparameters
    for lr in config["lr_list"]:
        for batch_size in config["batch_size_list"]:
            for target_layer in config["target_layer_list"]:
                for operation in config["operation_list"]:
                    for mask_init in config["mask_init_list"]:
                        for model_seed in config["model_seed_list"]:
                            # Create a new model_id
                            model_id = str(uuid.uuid4())

                            config["lr"] = lr
                            config["batch_size"] = batch_size

                            config["target_layer"] = target_layer
                            config["operation"] = operation
                            config["mask_init_value"] = mask_init
                            
                            # Set Model seed
                            torch.manual_seed(model_seed)
                            np.random.seed(model_seed)
                            model, tokenizer = utils.get_model_and_tokenizer(config)

                            # Flag for setting tokenizer length on multiberts models
                            if "multiberts" in tokenizer.name_or_path:
                                tokenizer.model_max_length = 512

                            # Force datasets to be the same
                            torch.manual_seed(config["data_seed"])
                            np.random.seed(config["data_seed"])
                            trainloader, devloader, testloader = utils.create_datasets(
                                config, tokenizer
                            )

                            train_length = get_dataset_length(trainloader)
                            dev_length = get_dataset_length(devloader)
                            test_length = get_dataset_length(testloader)

                            probe = utils.create_circuit_probe(config, model, tokenizer)
                            probe.to(device)

                            if config["save_models"]:
                                os.makedirs(
                                    os.path.join(config["model_dir"], model_id), exist_ok=True
                                )
                            os.makedirs(config["results_dir"], exist_ok=True)

                            if config["num_epochs"] != 0:
                                probe = train_probe(config, probe, trainloader, devloader)
                            final_train_loss = eval_probe(config, probe, trainloader)
                            final_eval_loss = eval_probe(config, probe, devloader)

                            knn_results = knn_evaluation(config, probe, trainloader, testloader)

                            l0_statistics = probe.wrapped_model.model.compute_l0_statistics()

                            output_dict = {
                                "model_id": [model_id],
                                "label": [config["label"]],
                                "train length": [train_length],
                                "dev length": [dev_length],
                                "test length": [test_length],
                                "batch_size": [config["batch_size"]],
                                "lr": [config["lr"]],
                                "data_seed": [config["data_seed"]],
                                "model_seed": [model_seed],
                                "target_layer": [config["target_layer"]],
                                "operation": [config["operation"]],
                                "mask_init": [mask_init],
                                "model": [config["model_type"]],
                                "model path": [config["model_path"]],
                                "train loss": [final_train_loss],
                                "dev loss": [final_eval_loss],
                                "knn dev acc": [knn_results["dev_acc"]],
                                "dev majority acc": [knn_results["dev_majority"]],
                                "knn test acc": [knn_results["test_acc"]],
                                "test majority acc": [knn_results["test_majority"]],
                                "L0 Norm": [l0_statistics["total_l0"].cpu().item()],
                                "L0 Max": [l0_statistics["max_l0"]],
                            }

                            if config["lm_eval"] == True:
                                probe.train(False)
                                model = probe.wrapped_model.model
                                model.set_ablate_mode("zero_ablate")
                                lm_loader, mlm_loader = get_lm_eval_data(config, tokenizer)
                                lm_results = lm_eval(config, model, tokenizer, lm_loader)
                                mlm_results = masked_lm_eval(
                                    config, model, tokenizer, mlm_loader
                                )

                                output_dict["vanilla acc"] = [lm_results["vanilla_acc"]]
                                output_dict["ablated acc"] = [lm_results["ablated_acc"]]
                                output_dict["kl"] = [lm_results["kl"]]
                                output_dict["mlm vanilla acc"] = [
                                    mlm_results["mlm_vanilla_acc"]
                                ]
                                output_dict["mlm ablated acc"] = [
                                    mlm_results["mlm_ablated_acc"]
                                ]
                                output_dict["mlm kl"] = [mlm_results["mlm_kl"]]

                                if config["num_epochs"] != 0:
                                    random_ablated_kls = []
                                    random_ablated_mlm_kls = []                            
                                    for _ in range(config["num_random_ablations"]):
                                        model.set_ablate_mode("randomly_sampled", force_resample=True)
                                        random_ablate_lm_results = lm_eval(config, model, tokenizer, lm_loader)
                                        random_ablate_mlm_results = masked_lm_eval(
                                            config, model, tokenizer, mlm_loader
                                        )
                                        random_ablated_kls.append(random_ablate_lm_results["kl"])
                                        random_ablated_mlm_kls.append(random_ablate_mlm_results["mlm_kl"])

                                    output_dict["random ablate kl mean"] = [np.mean(random_ablated_kls)]
                                    output_dict["random ablate kl std"] = [np.std(random_ablated_kls)]
                                    output_dict["all random ablate kls"] = [str(random_ablated_kls)]

                                    output_dict["random ablate mlm kl mean"] = [np.mean(random_ablated_mlm_kls)]
                                    if np.mean(random_ablated_mlm_kls) == -1:
                                        output_dict["random ablate mlm kl std"] = [-1]
                                    else:
                                        output_dict["random ablate mlm kl std"] = [np.std(random_ablated_mlm_kls)]
                                    output_dict["all random ablate mlm kls"] = [str(random_ablated_mlm_kls)]

                                model.set_ablate_mode("none")


                            df = pd.concat(
                                [df, pd.DataFrame.from_dict(output_dict)], ignore_index=True
                            )

                            print("Saving csv")
                            # Will overwrite this file after every evaluation
                            df.to_csv(os.path.join(config["results_dir"], "results.csv"))

                            if config["save_models"]:
                                torch.save(
                                    probe.state_dict(),
                                    os.path.join(config["model_dir"], model_id),
                                )


if __name__ == "__main__":
    main()
