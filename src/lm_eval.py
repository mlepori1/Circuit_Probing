import torch
import torch.nn.functional as F
import numpy as np
from LMEvalDataset import LMEvalDataset, MLMEvalDataset
from torch.utils.data import random_split, DataLoader


def get_lm_eval_data(config, tokenizer):
    # Gets data for running behavioral lm evals
    lm_dataset = LMEvalDataset(config["lm_data_path"], tokenizer, seed=config["seed"])
    mlm_dataset = MLMEvalDataset(config["lm_data_path"], tokenizer, seed=config["seed"])

    remainder = len(lm_dataset) - config["lm_size"]
    lm_dataset, _ = random_split(lm_dataset, [config["lm_size"], remainder])
    mlm_dataset, _ = random_split(mlm_dataset, [config["lm_size"], remainder])

    lm_loader = DataLoader(lm_dataset, batch_size=config["batch_size"])
    mlm_loader = DataLoader(mlm_dataset, batch_size=1)  # Batch size MUST be 1 for this
    return lm_loader, mlm_loader


def lm_eval(config, model, tokenizer, dataloader):
    # Runs eval for MLM with no masked entries
    kl_divs = []
    abl_correct = []
    vanilla_correct = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        abl_outputs = model(**batch).logits
        abl_outputs = abl_outputs.reshape(
            -1, abl_outputs.shape[-1]
        )  # Reshape outputs into [# Tokens, Vocab Size]

        model.use_masks(False)
        vanilla_outputs = model(**batch).logits
        vanilla_outputs = vanilla_outputs.reshape(-1, abl_outputs.shape[-1])
        model.use_masks(True)

        # Compute Accuracies
        labels = batch["input_ids"].reshape(-1)
        not_special_tokens = ~torch.tensor(
            tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
        ).bool()

        if config["model_type"] == "gpt2":
            # Shift labels right one place
            label_mask = (
                torch.cat((torch.tensor([0]), not_special_tokens[:-1]))
                .reshape(-1)
                .bool()
            )
        else:
            label_mask = not_special_tokens

        labels = labels[label_mask]

        # Only account for non-special tokens
        abl_outputs = abl_outputs[not_special_tokens]
        vanilla_outputs = vanilla_outputs[not_special_tokens]

        abl_preds = torch.argmax(abl_outputs, dim=-1)
        van_preds = torch.argmax(vanilla_outputs, dim=-1)

        abl_correct += list((abl_preds == labels).cpu())
        vanilla_correct += list((van_preds == labels).cpu())

        # Compute KL Div
        kl_divs += list(
            F.kl_div(
                F.log_softmax(abl_outputs, dim=-1),
                F.log_softmax(vanilla_outputs, dim=-1),
                log_target=True,
                reduction="none",
            )
            .sum(dim=-1)
            .cpu()
        )

    return {
        "vanilla_acc": np.sum(vanilla_correct) / len(vanilla_correct),
        "ablated_acc": np.sum(abl_correct) / len(abl_correct),
        "kl": np.sum(kl_divs) / len(kl_divs),
    }


def masked_lm_eval(config, model, tokenizer, dataloader):
    # Runs LM eval for mlm with 1 masked token per datapoint
    if config["model_type"] == "gpt2":
        return {"mlm_vanilla_acc": -1, "mlm_ablated_acc": -1, "mlm_kl": -1}

    kl_divs = []
    abl_correct = []
    vanilla_correct = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        input_ids = batch["input_ids"][0]  # Batching happens within dataset
        abl_outputs = model(input_ids=input_ids).logits
        abl_outputs = abl_outputs.reshape(
            -1, abl_outputs.shape[-1]
        )  # Reshape outputs into [# Tokens, Vocab Size]

        model.use_masks(False)
        vanilla_outputs = model(input_ids=input_ids).logits
        vanilla_outputs = vanilla_outputs.reshape(-1, abl_outputs.shape[-1])
        model.use_masks(True)

        # Compute Accuracies
        labels = batch["labels"].reshape(-1)
        masked_ids = (input_ids == tokenizer.mask_token_id).reshape(-1)

        # Only account for masked tokens
        abl_outputs = abl_outputs[masked_ids]
        vanilla_outputs = vanilla_outputs[masked_ids]
        labels = labels[masked_ids]

        abl_preds = torch.argmax(abl_outputs, dim=-1)
        van_preds = torch.argmax(vanilla_outputs, dim=-1)

        abl_correct += list((abl_preds == labels).cpu())
        vanilla_correct += list((van_preds == labels).cpu())

        # Compute KL Div
        kl_divs += list(
            F.kl_div(
                F.log_softmax(abl_outputs, dim=-1),
                F.log_softmax(vanilla_outputs, dim=-1),
                log_target=True,
                reduction="none",
            )
            .sum(dim=-1)
            .cpu()
        )

    return {
        "mlm_vanilla_acc": np.sum(vanilla_correct) / len(vanilla_correct),
        "mlm_ablated_acc": np.sum(abl_correct) / len(abl_correct),
        "mlm_kl": np.sum(kl_divs) / len(kl_divs),
    }
