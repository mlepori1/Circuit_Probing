import copy
import os
import shutil
from functools import partial

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

import data_utils
import utils

# A script that generates arithmetic data and saves it
# then trains a small GPT2 style transformer on that dataset
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
    for fn_def in string_fns:
        fn = fn_def[0]
        if "mod" in fn:
            modulo = int(fn.split("_")[-1])
            fn = "_".join(fn.split("_")[:-1])
            intermediate_functions.append((partial(STR_2_FN[fn], p=modulo), fn_def[1], fn_def[2]))
        else:
            intermediate_functions.append((STR_2_FN[fn], fn_def[1], fn_def[2]))
    return intermediate_functions


def train_loop(model, train_x, train_y, test_x, test_y, lr, config):
    # A simple train loop
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

    # Create Datasets using helper functions
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

        if "counterfactual_label_functions" in config.keys():
            counterfactual_label_functions = config["counterfactual_label_functions"]
        else:
            counterfactual_label_functions = []

        torch.manual_seed(config["data_seed"])
        train_x, train_y, test_x, test_y = data_utils.generate_data(
            intermediate_functions,
            config["mod"],
            train_frac=config["train_frac"],
            device=config["device"],
            data_path=config["data_dir"],
            auxiliary_variable_functions=auxiliary_functions,
            counterfactual_label_functions=counterfactual_label_functions,
        )
    else:
        intermediate_functions_1 = convert_strings_to_functions(
            config["intermediate_functions_1"]
        )
        intermediate_functions_2 = convert_strings_to_functions(
            config["intermediate_functions_2"]
        )

        if "task_1_auxiliary_functions" in config.keys():
            task_1_auxiliary_functions = convert_strings_to_functions(
                config["task_1_auxiliary_functions"]
            )
        else:
            task_1_auxiliary_functions = []

        if "task_2_auxiliary_functions" in config.keys():
            task_2_auxiliary_functions = convert_strings_to_functions(
                config["task_2_auxiliary_functions"]
            )
        else:
            task_2_auxiliary_functions = []

        torch.manual_seed(config["data_seed"])
        train_x, train_y, test_x, test_y = data_utils.generate_multitask_data(
            intermediate_functions_1,
            intermediate_functions_2,
            config["mod"],
            train_frac=config["train_frac"],
            device=config["device"],
            data_path=config["data_dir"],
            task_1_aux_functions=task_1_auxiliary_functions,
            task_2_aux_functions=task_2_auxiliary_functions,
            task_1_counterfactual_label_functions=config["task_1_counterfactual_label_functions"],
            task_2_counterfactual_label_functions=config["task_2_counterfactual_label_functions"],
        )

    # Iterate through all training hyperparameters
    for lr in config["lr_list"]:
        for model_seed in config["model_seed_list"]:
            # Create a new model_id
            model_id = "model_LR_" + str(lr) + "_Seed_" + str(model_seed)
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

            # If we are doing transfer learning, then load the model state from a saved state_dict
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

            plt.figure()
            sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
            lineplot_loss_df = pd.DataFrame({
                'Epochs': np.arange(0, len(train_losses) + 1, config["checkpoint_every"]),
                'Train Loss': train_losses[indices],
                'Test Loss': test_losses
            })
            ax = sns.lineplot(
                x="Epochs", y="Loss", hue="Dataset", data=pd.melt(lineplot_loss_df, ['Epochs'], value_name="Loss", var_name="Dataset")
                ).set(title=config["figtitle"] + " Loss")
            plt.savefig(os.path.join(config["results_dir"], "Loss.pdf"), format="pdf", bbox_inches="tight")

            # Save the accuracy plot for train and test
            plt.figure()
            sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
            lineplot_acc_df = pd.DataFrame({
                'Epochs': np.arange(0, len(train_losses) + 1, config["checkpoint_every"]),
                'Train Acc': train_accs[indices],
                'Test Acc': test_accs
            })
            ax = sns.lineplot(
                x="Epochs", y="Accuracy", hue="Dataset", data=pd.melt(lineplot_acc_df, ['Epochs'], value_name="Accuracy", var_name="Dataset")
                ).set(title=config["figtitle"] + " Accuracy")
            plt.savefig(os.path.join(config["results_dir"], "Acc.pdf"), format="pdf", bbox_inches="tight")

            # Transfer vs. Reinitialized Plots
            if config["transfer"]:

                plt.figure()
                sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
                lineplot_loss_df = pd.DataFrame({
                    'Epochs': np.arange(0, len(transfer_train_losses) + 1, config["checkpoint_every"]),
                    'Transfer Train Loss': transfer_train_losses[indices],
                    'Transfer Test Loss': transfer_test_losses,
                    'Reinit Train Loss': train_losses[indices],
                    'Reinit Test Loss': test_losses
                })
                ax = sns.lineplot(
                    x="Epochs", y="Loss", hue="Dataset", data=pd.melt(lineplot_loss_df, ['Epochs'], value_name="Loss", var_name="Dataset")
                    ).set(title=config["figtitle"] + " Transfer vs. Reinitialized Loss")
                plt.savefig(os.path.join(config["results_dir"], "Transfer_Loss.pdf"), format="pdf", bbox_inches="tight")

                plt.figure()
                sns.set(style="darkgrid", palette="Dark2", font_scale=1.25)
                lineplot_acc_df = pd.DataFrame({
                    'Epochs': np.arange(0, len(transfer_train_losses) + 1, config["checkpoint_every"]),
                    'Transfer Train Acc': transfer_train_accs[indices],
                    'Transfer Test Acc': transfer_test_accs,
                    'Reinit Train Acc': train_accs[indices],
                    'Reinit Test Acc': test_accs
                })
                ax = sns.lineplot(
                    x="Epochs", y="Accuracy", hue="Dataset", data=pd.melt(lineplot_acc_df, ['Epochs'], value_name="Accuracy", var_name="Dataset")
                    ).set(title=config["figtitle"] + " Transfer vs. Reinitialized Accuracy")
                plt.savefig(os.path.join(config["results_dir"], "Transfer_Acc.pdf"), format="pdf", bbox_inches="tight")


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
