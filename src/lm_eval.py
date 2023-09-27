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
    # Get accuracy of model before and after ablation
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
    # Run model evaluation with SV-Agreement data. Check whether logit for 
    # "is" > logit for "are" to see whether model distinguishes syntactic number of subject 
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

def agreement_qualitative_eval(config, outpath, model, tokenizer, dataloader, ablate_set=None):
    # Record the top 50 next word predictions for a batch of data before and after ablation
    batch = next(iter(dataloader))
    batch = {k: v.to(config["device"]) for k, v in batch.items()}

    model.use_masks(False)
    vanilla_outputs = model(input_ids=batch["input_ids"]).logits
    vanilla_outputs = vanilla_outputs[batch["token_mask"]]
    _, vanilla_indices = torch.topk(vanilla_outputs, k=50, dim=-1)

    model.use_masks(True, ablate_set)
    abl_outputs = model(input_ids=batch["input_ids"]).logits
    abl_outputs = abl_outputs[batch["token_mask"]]
    _, abl_indices = torch.topk(abl_outputs, k=50, dim=-1)

    inputs = tokenizer.batch_decode(batch["input_ids"])

    lines = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        vanilla_pred = "\n".join(tokenizer.convert_ids_to_tokens(vanilla_indices[i]))
        abl_pred = "\n".join(tokenizer.convert_ids_to_tokens(abl_indices[i]))
        lines.append(f"Input: {ipt}\nTop 50 Model Predictions:\n****************\n{vanilla_pred}\nTop 50 Ablated Model Predictions:\n*****************\n{abl_pred}")
    lines = "\n".join(lines)
    file = open(outpath, "w")
    file.write(lines)
    file.close()

def reflexive_eval(config, model, dataloader, ablate_set=None):
    # Model evaluation before and after ablating reflexives subnetwork
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