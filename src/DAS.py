from utils import get_config
from align_transformers.models.modelings_alignable import AutoAlignableModel
from align_transformers.trainer import Aligner
from torch.utils.data import DataLoader, Dataset, random_split
from Datasets import DASDataset
import pandas as pd
import torch
import os
from transformers import (
    get_linear_schedule_with_warmup,
    logging
)


def convert_datafile_to_das_dataloader(path, cf_label, token_range, batch_size, n_train_examples=None, n_val_examples=None):
    data = pd.read_csv(path)
    dataset = DASDataset(data, cf_label, token_range, "cuda")
    if n_train_examples is not None:
        remainder = len(dataset) - (n_train_examples + n_val_examples)

        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        train_data, eval_data, _ = random_split(
            dataset,
            [n_train_examples, n_val_examples, remainder],
            generator=generator,
        )
        return DataLoader(train_data, batch_size=batch_size), DataLoader(eval_data, batch_size=batch_size)
    else:
        return DataLoader(dataset, batch_size=batch_size)
    

if __name__ == '__main__':

    ## Default parameters from demo
    batch_size = 128
    rotation_lr = 0.01
    boundaries_lr = 5e-3
    seed = 0
    gradient_accumulation_steps = 1
    log_step = 10
    valid_steps = 500 # validation steps
    logger = logging.get_logger("transformers")

    ## Custom parameters
    config = get_config()
    epochs = config["epochs"]
    output_dir = config["results_dir"]
    token_position_strategy = config["token_range"] # Final token of [a, b, p]
    token_range = [int(i) for i in token_position_strategy.split("_")]
    n_train_examples = config["n_train"]
    n_val_examples = config["n_val"]

    for location in config["location_list"]:
        ## Construct Alignable Model
        alignment_config = {
        'layer': 0,
        'num_of_das_token' : 1,
        'location': location
        }
        output_dir = config["results_dir"] + f"_{location}/"
        model = AutoAlignableModel.from_pretrained(config["model_path"], alignment_config)

        ## Create Datasets
        train_dataloader, eval_dataloader = convert_datafile_to_das_dataloader(config["train_path"], cf_label=config["counterfactual_label"], token_range=token_range, batch_size=batch_size, n_train_examples=n_train_examples, n_val_examples=n_val_examples)
        test_dataloader = convert_datafile_to_das_dataloader(config["test_path"], cf_label=config["counterfactual_label"], token_range=token_range, batch_size=batch_size)

        ## Metric
        def compute_metrics(eval_preds, eval_labels):
            total_count = 0
            correct_count = 0
            for eval_pred, eval_label in zip(eval_preds, eval_labels):
                actual_test_labels = eval_label[:, -1]
                pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
                correct_labels = (actual_test_labels==pred_test_labels)
                total_count += len(correct_labels)
                correct_count += correct_labels.sum().tolist()
            accuracy = round(correct_count/total_count, 2)
            print(accuracy)
            return {"accuracy" : accuracy}
        
        ## Training
        # set off the gradients among all other layers.
        for name, param in model.named_parameters():
            if "rotate_layer" not in name and "intervention_boundaries" not in name:
                param.requires_grad = False

        t_total = int(len(train_dataloader) * epochs)
        warm_up_steps = 0.1 * t_total
        optimizer = torch.optim.Adam(
            [{'params': model.transformer.rotate_layer.parameters()},
            {'params': model.transformer.intervention_boundaries, 'lr': boundaries_lr}],
            lr=rotation_lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_steps,
            num_training_steps=t_total
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.to(torch.device("cuda")) # no rank is needed!

        aligner = Aligner(
            model,
            logger=logger,
            is_wandb=False,
            is_master=True,
            n_gpu=1,
            model_name="GPT2LMHeadModel",
            device="cuda",
            compute_metrics=compute_metrics
        )

        # Train
        aligner.train(
            train_dataloader, 
            test_dataloader, 
            test_dataloader,
            optimizer, 
            scheduler, 
            log_step=log_step, 
            valid_steps=valid_steps,
            output_dir=output_dir, 
            epochs=epochs, 
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
