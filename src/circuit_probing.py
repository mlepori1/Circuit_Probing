import utils
import uuid
import os
import numpy as np
import shutil
import torch
import pandas as pd
import os

from transformers import Trainer, TrainerCallback, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer
import transformers


class TemperatureCallback(TrainerCallback):

    def __init__(self, total_epochs, final_temp):
        self.temp_increase = final_temp**(1./total_epochs)

    def on_evaluate(self, args, state, control, **kwargs):
            model = kwargs.pop('model')
            temp = model.get_temp()
            model.set_temp(temp * self.temp_increase)

class L0MetricsCallback(TrainerCallback):

    def __init__(self):
        super().__init__()

    def get_l0_norm(self, model):
        masks = []
        for layer in model.modules():
            if hasattr(layer, "mask_weight"):
                masks.append(layer.mask.detach().cpu())
        l0_norm = int(sum(m.sum() for m in masks))
        l0_max =sum([len(m.reshape(-1)) for m in masks])
        return l0_norm, l0_max

    def on_log(self, args, state, control, **kwargs):
        l0_norm, l0_max = self.get_l0_norm(kwargs.pop('model'))
        recent_log = state.log_history[-1]
        recent_log["L0 Norm"] = l0_norm
        recent_log["L0 Max"] = l0_max
        state.log_history[-1] = recent_log

    def on_evaluate(self, args, state, control, **kwargs):
        self.l0_norm, self.l0_max = self.get_l0_norm(kwargs.pop('model'))
        print("L0 Norm, L0 Max:")
        print(self.l0_norm)
        print(self.l0_max)

def main():

    config = utils.get_config()
    tokenizer = AutoTokenizer(config["tokenizer_path"])
    trainset, valset, testset = utils.load_dataset(config, tokenizer)

    df = pd.DataFrame()

    # Iterate through all training hyperparameters
    for seed in config["seed_list"]:
        for lr in config["lr_list"]:
            for batch_size in config["batch_size_list"]:
                for target_stage in config["target_stage_list"]:
                    for mask_init in config["mask_init_list"]:

                        # Create a new model_id
                        model_id = str(uuid.uuid4())

                        config["lr"] = lr
                        config["batch_size"] = batch_size

                        config["seed"] = seed
                        if config["seed"] is not None:
                            transformers.set_seed(config["seed"])

                        config["l0_start"] = target_stage
                        config["l0_end"] = target_stage + 1
                        config["mask_init_value"] = mask_init

                        model = utils.create_model(tokenizer, config)

                        os.makedirs(os.path.join(config["model_dir"], model_id), exist_ok=True)
                        os.makedirs(config["results_dir"], exist_ok=True)

                        # Set up callbacks
                        callbacks = []
                        l0_metrics = L0MetricsCallback()
                        callbacks.append(l0_metrics)
                        callbacks.append(TensorBoardCallback())
                        callbacks.append(TemperatureCallback(config["max_epochs"], config["max_temp"]))

                        if config["early_stopping"]!=0:
                            callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["es_patience"]))
                    
                        training_args = utils.get_training_args(config, model_id)

                        trainer = Trainer(
                            model=model,
                            tokenizer=tokenizer, 
                            args=training_args, 
                            train_dataset=trainset, 
                            eval_dataset=valset, 
                            callbacks=callbacks
                            )
                        
                        trainer.train()
                        epoch = trainer.state.log_history[-1]["epoch"]
                        
                        train_results = trainer.evaluate(trainset)
                        val_results = trainer.evaluate(valset)
                        test_results = trainer.evaluate(testset)

                        l0_norm, l0_max = l0_metrics.get_l0_norm(trainer.model)

                        output_dict = {
                                'model_id': model_id,
                                'task': config["task"],
                                "seed": seed,
                                'batch_size': config["batch_size"],
                                'lr': config["lr"],
                                "epoch": epoch,
                                "l0 start": target_stage,
                                "mask_init": mask_init,
                                "pt_weights": config["pretrained_weights"],
                                'train loss': train_results["eval_loss"],
                                'eval loss': val_results["eval_loss"],
                                'test loss': test_results["eval_loss"],
                                'L0 Norm': l0_norm,
                                'L0 Max': l0_max
                            }

                        df = df.append(output_dict, ignore_index=True)

                        print("Saving csv")
                        # Will overwrite this file after every evaluation
                        df.to_csv(os.path.join(config["results_dir"], 'results.csv'))

                        # Get rid of checkpoint dirs after testing
                        shutil.rmtree(os.path.join(config["model_dir"], model_id))

                        if config["save_models"]:
                            trainer.save_model(os.path.join(config["model_dir"], model_id))

if __name__ == "__main__":
    main()