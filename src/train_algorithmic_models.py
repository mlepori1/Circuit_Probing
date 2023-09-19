import copy
import os
import shutil
import uuid
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

import data_utils
import utils


def loss_fn(logits, labels):
    logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def compute_acc(logits, labels):
    logits = logits[:, -1]
    logits = logits.to(torch.float64)
    predictions = logits.argmax(dim=-1)
    return (torch.sum(predictions == labels) / len(labels)).cpu()


def convert_strings_to_functions(string_fns):
    intermediate_functions = []
    for fn in string_fns:
        if "mod" in fn:
            modulo = int(fn.split("_")[-1])
            fn = "_".join(fn.split("_")[:-1])
            intermediate_functions.append(partial(STR_2_FN[fn], p=modulo))
        else:
            intermediate_functions.append(STR_2_FN[fn])
    return intermediate_functions


def train_loop(model, train_x, train_y, test_x, test_y, lr, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config["wd"], betas=config["betas"]
    )
    scheduler = LinearLR(optimizer, total_iters=10)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    model_checkpoints = []
    checkpoint_epochs = []

    with torch.inference_mode():
        test_logits = model(test_x).logits
        test_loss = loss_fn(test_logits, test_y)
        test_losses.append(test_loss.cpu().item())
        test_accs.append(compute_acc(test_logits, test_y))

        checkpoint_epochs.append(0)
        model_checkpoints.append(copy.deepcopy(model.state_dict()))

    for epoch in tqdm(range(config["num_epochs"])):
        train_logits = model(train_x).logits
        train_loss = loss_fn(train_logits, train_y)
        train_loss.backward()
        train_losses.append(train_loss.cpu().item())
        train_accs.append(compute_acc(train_logits, train_y))

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if ((epoch + 1) % config["checkpoint_every"]) == 0:
            with torch.inference_mode():
                test_logits = model(test_x).logits
                test_loss = loss_fn(test_logits, test_y)
                test_losses.append(test_loss.cpu().item())
                test_accs.append(compute_acc(test_logits, test_y))

                checkpoint_epochs.append(epoch + 1)
                model_checkpoints.append(copy.deepcopy(model.state_dict()))
                print(
                    f"Epoch {epoch} Train Loss {train_losses[-1]} Test Loss {test_losses[-1]} Train Acc {train_accs[-1]} Test Acc {test_accs[-1]}"
                )

    return (
        model,
        np.array(train_losses),
        np.array(train_accs),
        np.array(test_losses),
        np.array(test_accs),
        model_checkpoints,
        checkpoint_epochs,
    )


STR_2_FN = {
    "a_identity": data_utils.a_identity,
    "a2": data_utils.a2,
    "a4": data_utils.a4,
    "a_mod": data_utils.a_mod,
    "b_identity": data_utils.b_identity,
    "b2": data_utils.b2,
    "minus_b2": data_utils.minus_b2,
    "b4": data_utils.b4,
    "b_mod": data_utils.b_mod,
    "ab": data_utils.ab,
    "a_plus_b": data_utils.a_plus_b,
    "a_minus_b": data_utils.a_minus_b,
    "a_plus_b_no_prime": data_utils.a_plus_b_no_prime,
    "a_minus_b_no_prime": data_utils.a_minus_b_no_prime,
}


def main():
    config = utils.get_config()
    df = pd.DataFrame()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device

    # Create Datasets
    shutil.rmtree(config["data_dir"], ignore_errors=True)
    os.makedirs(config["data_dir"], exist_ok=True)

    if config["multitask"] == False:
        intermediate_functions = convert_strings_to_functions(
            config["intermediate_functions"]
        )
        if "auxiliary_functions" in config.keys():
            auxiliary_functions = convert_strings_to_functions(
                config["auxiliary_functions"]
            )
        else:
            auxiliary_functions = []

        train_x, train_y, test_x, test_y = data_utils.generate_data(
            intermediate_functions,
            config["mod"],
            train_frac=config["train_frac"],
            data_seed=config["data_seed"],
            device=config["device"],
            data_path=config["data_dir"],
            auxiliary_variable_functions=auxiliary_functions,
            counterfactual_label_functions=config["counterfactual_label_functions"],
        )
    else:
        intermediate_functions_1 = convert_strings_to_functions(
            config["intermediate_functions_1"]
        )
        intermediate_functions_2 = convert_strings_to_functions(
            config["intermediate_functions_2"]
        )

        train_x, train_y, test_x, test_y = data_utils.generate_multitask_data(
            intermediate_functions_1,
            intermediate_functions_2,
            config["mod"],
            train_frac=config["train_frac"],
            data_seed=config["data_seed"],
            device=config["device"],
            save_data=True,
            data_path=config["data_dir"],
        )

    # Iterate through all training hyperparameters
    for lr in config["lr_list"]:
        for model_seed in config["model_seed_list"]:
            # Create a new model_id
            model_id = str(uuid.uuid4())

            config["lr"] = lr

            # Set Model seed
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)

            # Vocab size adds task tokens if multitasks
            vocab_size = (
                config["mod"] + 1 if config["multitask"] == False else config["mod"] + 3
            )
            cfg = GPT2Config(
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                n_positions=config["n_positions"],
                n_embd=config["n_embd"],
                n_inner=config["n_inner"],
                activation_function="relu",
                vocab_size=vocab_size,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
            model = GPT2LMHeadModel(cfg).to(config["device"])
            (
                model,
                train_losses,
                train_accs,
                test_losses,
                test_accs,
                model_checkpoints,
                checkpoint_epochs,
            ) = train_loop(model, train_x, train_y, test_x, test_y, lr, config)

            if config["save_models"]:
                shutil.rmtree(config["model_dir"], ignore_errors=True)
                os.makedirs(os.path.join(config["model_dir"], model_id), exist_ok=True)

            if config["transfer"]:
                transfer_model = GPT2LMHeadModel.from_pretrained(
                    config["model_path"]
                ).to(config["device"])
                (
                    transfer_model,
                    transfer_train_losses,
                    transfer_train_accs,
                    transfer_test_losses,
                    transfer_test_accs,
                    _,
                    _,
                ) = train_loop(
                    transfer_model, train_x, train_y, test_x, test_y, lr, config
                )

            shutil.rmtree(config["results_dir"], ignore_errors=True)
            os.makedirs(config["results_dir"], exist_ok=True)

            output_dict = {
                "model_id": [model_id],
                "train length": [len(train_x)],
                "test length": [len(test_x)],
                "lr": [config["lr"]],
                "data_seed": [config["data_seed"]],
                "model_seed": [model_seed],
                "train loss": [train_losses[-1]],
                "test loss": [test_losses[-1]],
                "train acc": [train_accs[-1].item()],
                "test acc": [test_accs[-1].item()],
                "model_dir": [config["model_dir"]],
            }

            df = pd.concat([df, pd.DataFrame.from_dict(output_dict)], ignore_index=True)
            print("Saving csv")
            # Will overwrite this file after every evaluation
            df.to_csv(os.path.join(config["results_dir"], "results.csv"))

            # Save the loss plot for train and test
            indices = np.array(range(0, len(train_losses) + 1, config["checkpoint_every"])) - 1
            indices[0] = 0
            plt.plot(
                np.arange(0, len(train_losses) + 1, config["checkpoint_every"]),
                train_losses[indices],
                label="train loss",
            )
            plt.plot(
                np.arange(0, len(train_losses) + 1, config["checkpoint_every"]),
                test_losses,
                label="test loss",
            )
            plt.title(str(model_id))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            if config["transfer"]:
                transfer_indices = np.array(range(0, len(transfer_train_losses) + 1, config["checkpoint_every"])) - 1
                transfer_indices[0] = 0
                plt.plot(
                    np.arange(
                        0, len(transfer_train_losses) + 1, config["checkpoint_every"]
                    ),
                    transfer_train_losses[indices],
                    label="transfer train loss",
                )
                plt.plot(
                    np.arange(
                        0, len(transfer_train_losses) + 1, config["checkpoint_every"]
                    ),
                    transfer_test_losses,
                    label="transfer test loss",
                )
                plt.title("Transfer Loss and Reinitialized Loss")
            plt.legend()
            plt.savefig(
                os.path.join(config["results_dir"], str(model_id) + "_loss.png")
            )

            # Save the accuracy plot for train and test
            plt.figure()
            plt.plot(
                np.arange(0, len(train_accs) + 1, config["checkpoint_every"]),
                train_accs[indices],
                label="train acc",
            )
            plt.plot(
                np.arange(0, len(train_accs) + 1, config["checkpoint_every"]),
                test_accs,
                label="test acc",
            )
            plt.title(str(model_id))
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            if config["transfer"]:
                plt.plot(
                    np.arange(0, len(transfer_train_accs) + 1, config["checkpoint_every"]),
                    transfer_train_accs[transfer_indices],
                    label="transfer train acc",
                )
                plt.plot(
                    np.arange(0, len(transfer_train_accs) + 1, config["checkpoint_every"]),
                    transfer_test_accs,
                    label="transfer test acc",
                )
                plt.title("Transfer Acc and Reinitialized Acc")
            plt.legend()
            plt.savefig(os.path.join(config["results_dir"], str(model_id) + "_acc.png"))

            if config["save_models"]:
                # Save off the earliest model that achieves perfect accuracy on train
                # This should be a memorized solution
                for ep in checkpoint_epochs:
                    if ep in config["save_checkpoints"]:
                        cp = model_checkpoints[checkpoint_epochs.index(ep)]
                        model.load_state_dict(cp)
                        model.save_pretrained(
                            os.path.join(config["model_dir"], model_id + "_" + str(ep))
                        )

                # Save final model
                # This should be a generalizable solution
                cp = model_checkpoints[-1]
                model.load_state_dict(cp)
                model.save_pretrained(
                    os.path.join(config["model_dir"], model_id + "_final")
                )


if __name__ == "__main__":
    main()
