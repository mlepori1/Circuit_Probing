import utils
import uuid
import os
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm


def eval_probe(config, probe, dataloader):
    # Implements a simple probe evaluation loop
    probe.train(False)
    average_eval_loss = []
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        out = probe(**batch)
        average_eval_loss.append(out.loss.detach().item())
        correct += torch.sum(torch.argmax(out.logits, dim=-1) == out.labels)
        total += len(out.labels)
    probe.train(True)
    print(correct/total)
    return {
        "average loss" : np.sum(average_eval_loss) / len(average_eval_loss),
        "accuracy": (correct/total).item()
    }


def train_probe(config, probe, trainloader, testloader):
    # Implements a simple training loop that optimizes a probe
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
        eval_res = eval_probe(config, probe, testloader)


        progress_bar.set_postfix(
            {
                "Train Loss": round(train_loss, 4),
                "Eval Acc": round(eval_res["accuracy"], 4)
            }
        )

        probe.train(True)

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
                    for model_seed in config["model_seed_list"]:
                        # Create a new model_id
                        model_id = str(uuid.uuid4())

                        config["lr"] = lr
                        config["batch_size"] = batch_size

                        config["target_layer"] = target_layer
                        config["operation"] = operation
                        
                        # Set Model seed
                        torch.manual_seed(model_seed)
                        np.random.seed(model_seed)
                        model, tokenizer = utils.get_model_and_tokenizer(config)

                        # Force datasets to be the same
                        torch.manual_seed(config["data_seed"])
                        np.random.seed(config["data_seed"])
                        trainloader, devloader, testloader = utils.create_datasets(
                            config, tokenizer
                        )

                        train_length = get_dataset_length(trainloader)
                        dev_length = get_dataset_length(devloader)
                        test_length = get_dataset_length(testloader)

                        probe = utils.create_subnetwork_probe(config, model, tokenizer)
                        probe.to(device)

                        if config["save_models"]:
                            os.makedirs(
                                os.path.join(config["model_dir"], model_id), exist_ok=True
                            )
                        os.makedirs(config["results_dir"], exist_ok=True)

                        if config["num_epochs"] != 0:
                            probe = train_probe(config, probe, trainloader, devloader)
                        final_train_res = eval_probe(config, probe, trainloader)
                        final_eval_res = eval_probe(config, probe, devloader)

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
                            "model": [config["model_type"]],
                            "model path": [config["model_path"]],
                            "train loss": [final_train_res["average loss"]],
                            "dev loss": [final_eval_res["average loss"]],
                            "train acc": [final_train_res["accuracy"]],
                            "dev acc": [final_eval_res["accuracy"]],
                        }

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
