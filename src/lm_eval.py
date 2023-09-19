import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from Datasets import LMEvalDataset


def get_lm_eval_data(config, data_path):
    # Gets data for running behavioral lm evals
    lm_dataset = LMEvalDataset(data_path)
    lm_loader = DataLoader(
        lm_dataset, config["batch_size"], shuffle=False, drop_last=False
    )
    return lm_loader


def lm_eval(config, model, dataloader, ablate_set=None):
    abl_correct = []
    vanilla_correct = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}

        # Get full model accuracy
        model.use_masks(False)
        vanilla_outputs = model(input_ids=batch["input_ids"]).logits
        vanilla_outputs = vanilla_outputs[:, -1]
        vanilla_outputs = vanilla_outputs.to(torch.float64)

        # Get ablated model accuracy
        model.use_masks(True, ablate_set)
        abl_outputs = model(input_ids=batch["input_ids"]).logits
        abl_outputs = abl_outputs[:, -1]
        abl_outputs = abl_outputs.to(torch.float64)

        # Compute accuracy
        abl_preds = torch.argmax(abl_outputs, dim=-1)
        van_preds = torch.argmax(vanilla_outputs, dim=-1)

        abl_correct += list((abl_preds == batch["labels"]).cpu())
        vanilla_correct += list((van_preds == batch["labels"]).cpu())

    model.use_masks(True)

    return {
        "vanilla_acc": np.sum(vanilla_correct) / len(vanilla_correct),
        "ablated_acc": np.sum(abl_correct) / len(abl_correct),
    }


def agreement_eval(config, model, dataloader, sing_id, plur_id, ablate_set=None):
    abl_correct = []
    vanilla_correct = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}

        # Get full model accuracy
        model.use_masks(False)
        vanilla_outputs = model(input_ids=batch["input_ids"]).logits
        vanilla_outputs = vanilla_outputs[batch["token_mask"]]
        vanilla_outputs = vanilla_outputs.to(torch.float64)

        # Get ablated model accuracy
        model.use_masks(True, ablate_set)
        abl_outputs = model(input_ids=batch["input_ids"]).logits
        abl_outputs = abl_outputs[batch["token_mask"]]
        abl_outputs = abl_outputs.to(torch.float64)

        # Compute singular vs. plural
        # 0 corresponds to singular, 1 to plural
        abl_preds = abl_outputs[:, plur_id] > abl_outputs[:, sing_id]
        van_preds = vanilla_outputs[:, plur_id] > vanilla_outputs[:, sing_id]

        abl_correct += list((abl_preds == batch["labels"]).cpu())
        vanilla_correct += list((van_preds == batch["labels"]).cpu())

    model.use_masks(True)

    return {
        "vanilla_acc": np.sum(vanilla_correct) / len(vanilla_correct),
        "ablated_acc": np.sum(abl_correct) / len(abl_correct),
    }

def reflexive_eval(config, model, dataloader, ablate_set=None):
    abl_correct = []
    vanilla_correct = []

    loss = nn.CrossEntropyLoss(reduce=False)
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}

        # Get full model accuracy
        model.use_masks(False)
        gram_outputs = model(input_ids=batch["input_ids"])

        # Get logits predicting pronoun
        logit_mask = batch["token_mask"] + torch.roll(batch["token_mask"], 1, -1)
        logits = gram_outputs.logits[logit_mask]
        # Get labels for the pronoun
        label_mask = torch.roll(logit_mask, 1, -1)
        labels = batch["input_ids"][label_mask]
  
        # Flatten the tokens
        gram_loss = loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        gram_loss = torch.sum(gram_loss.reshape(-1, 2), -1)

        ungram_outputs = model(input_ids=batch["ungrammatical"])
        # Get logits predicting pronoun
        logit_mask = batch["token_mask"] + torch.roll(batch["token_mask"], 1, -1)
        logits = ungram_outputs.logits[logit_mask]

        # Get labels for the pronoun
        label_mask = torch.roll(logit_mask, 1, -1)
        labels = batch["ungrammatical"][label_mask]

        # Flatten the tokens
        ungram_loss = loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        ungram_loss = torch.sum(ungram_loss.reshape(-1, 2), -1)

        vanilla_correct += list((ungram_loss > gram_loss).cpu())

        # Get ablated model accuracy
        model.use_masks(True, ablate_set)
        gram_outputs = model(input_ids=batch["input_ids"])
        
        # Get logits predicting pronoun
        logit_mask = batch["token_mask"] + torch.roll(batch["token_mask"], 1, -1)
        logits = gram_outputs.logits[logit_mask]
        # Get labels for the pronoun
        label_mask = torch.roll(logit_mask, 1, -1)
        labels = batch["input_ids"][label_mask]
  
        # Flatten the tokens
        gram_loss = loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        gram_loss = torch.sum(gram_loss.reshape(-1, 2), -1)

        ungram_outputs = model(input_ids=batch["ungrammatical"])
        # Get logits predicting pronoun
        logit_mask = batch["token_mask"] + torch.roll(batch["token_mask"], 1, -1)
        logits = ungram_outputs.logits[logit_mask]

        # Get labels for the pronoun
        label_mask = torch.roll(logit_mask, 1, -1)
        labels = batch["ungrammatical"][label_mask]

        # Flatten the tokens
        ungram_loss = loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        ungram_loss = torch.sum(ungram_loss.reshape(-1, 2), -1)
        abl_correct += list((ungram_loss > gram_loss).cpu())

    model.use_masks(True)

    return {
        "vanilla_acc": np.sum(vanilla_correct) / len(vanilla_correct),
        "ablated_acc": np.sum(abl_correct) / len(abl_correct),
    }