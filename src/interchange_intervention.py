import os
import shutil
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from transformers import GPT2LMHeadModel, GPT2Config
from transformer_lens import HookedTransformer, HookedTransformerConfig, loading_from_pretrained
import transformer_lens.utils as lens_utils

import utils
from Datasets import InterchangeInterventionDataset

def create_interchange_dataset(config):
    return InterchangeInterventionDataset(config["data_path"], config["counterfactual_label"], config["device"])

def get_interchange_model(config):
    hf_model = GPT2LMHeadModel.from_pretrained(config["model_path"]).to(config["device"])
    for param in hf_model.parameters():
        param.requires_grad = False
    
    hf_config = GPT2Config.from_pretrained(config["model_path"])
    cfg_dict = {
        "d_model": hf_config.n_embd,
        "d_head": hf_config.n_embd // hf_config.n_head,
        "n_heads": hf_config.n_head,
        "d_mlp": hf_config.n_inner,
        "n_layers": hf_config.n_layer,
        "n_ctx": hf_config.n_positions,
        "eps": hf_config.layer_norm_epsilon,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
        "normalization_type": "LN",
    }

    hooked_config = HookedTransformerConfig(**cfg_dict)
    state_dict = loading_from_pretrained.convert_gpt2_weights(hf_model, hooked_config)
    hooked_model = HookedTransformer(hooked_config)
    state_dict = loading_from_pretrained.fill_missing_keys(hooked_model, state_dict)
    hooked_model.load_state_dict(state_dict)
    return hooked_model

def residual_stream_patching_hook(
    resid_state,
    hook,
    position,
    counterfactual_cache,
):
    # Each HookPoint has a name attribute giving the name of the hook.
    counterfactual_resid = counterfactual_cache[hook.name]
    resid_state[:, position, :] = counterfactual_resid[:, position, :]
    return resid_state

def compute_logit_difference(
        patched_logits,
        untouched_logits,
        sample
):
    patched_diff = (patched_logits[sample["counterfactual_label"]] - patched_logits[sample["original_label"]]).cpu().item()
    untouched_diff = (untouched_logits[sample["counterfactual_label"]] - untouched_logits[sample["original_label"]]).cpu().item()
    return patched_diff/untouched_diff


def evaluate_interchange_intervention(config, model, dataset):
    logit_diffs = []
    counterfactual_correct = 0
    original_correct = 0
    total = 0
    progress_bar = tqdm(range(len(dataset)))

    for sample in dataset:
        _, counterfactual_cache = model.run_with_cache(sample["counterfactual_ids"])
        hook_fn = partial(residual_stream_patching_hook, position=config["target_position"], counterfactual_cache=counterfactual_cache)
        patched_logits = model.run_with_hooks(sample["original_ids"], fwd_hooks=[
            (lens_utils.get_act_name(config["residual_location"], config["target_layer"]), hook_fn)
        ])
        predicted_label = torch.argmax(patched_logits[0, -1])
        if predicted_label == sample["counterfactual_label"]:
            counterfactual_correct += 1

        untouched_logits = model(sample["original_ids"])
        untouched_predicted_label = torch.argmax(untouched_logits[0, -1])
        if untouched_predicted_label == sample["original_label"]:
            original_correct += 1

        if sample["original_label"] != sample["counterfactual_label"]:
            logit_difference = compute_logit_difference(patched_logits[0, -1], untouched_logits[0, -1], sample)
            logit_diffs.append(logit_difference)
        total += 1

        # Sanity check: in the last layer, the LM head predicts from the final token. 
        # Thus, the decision cannot be influenced by the residual stream state of the 
        # other tokens in the last layer. Patching must have no effect! 
        if config["target_layer"] == 1 and config["target_position"] < 2:
            assert torch.all(patched_logits[0, -1] == untouched_logits[0, -1])
            assert untouched_predicted_label == predicted_label

        logit_diff_mean = np.mean(logit_diffs)
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "Counterfactual Correct": round(counterfactual_correct/total, 4),
                "Original Correct": round(original_correct/total, 4),
                "Logit Difference": round(logit_diff_mean, 4),
            }
        )

    return counterfactual_correct, original_correct, logit_diff_mean



def main():

    torch.set_grad_enabled(False)
    config = utils.get_config()
    df = pd.DataFrame()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device
    shutil.rmtree(config["results_dir"], ignore_errors=True)
    os.makedirs(config["results_dir"])

    for target_layer in config["target_layer_list"]:
        for target_position in config["target_position_list"]:
            for residual_location in config["residual_location_list"]:

                print(f"Layer: {target_layer} Position: {target_position} Residual: {residual_location}")
                # Create a new model_id
                model_id = str(uuid.uuid4())

                config["target_layer"] = target_layer
                config["target_position"] = target_position
                config["residual_location"] = residual_location

                model = get_interchange_model(config).to(device)

                # Force datasets to be the same
                torch.manual_seed(config["data_seed"])
                np.random.seed(config["data_seed"])

                dataloader = create_interchange_dataset(
                        config
                    )
                    
                counterfactual_accuracy, original_model_accuracy, logit_diff = evaluate_interchange_intervention(config, model, dataloader)

                output_dict = {
                    "model_id": [model_id],
                    "data_seed": [config["data_seed"]],
                    "target_layer": [config["target_layer"]],
                    "target_position": [config["target_position"]],
                    "residual_location": [config["residual_location"]],
                    "model path": [config["model_path"]],
                    "counterfactual accuracy": [counterfactual_accuracy],
                    "original model accuracy": [original_model_accuracy],
                    "logit diff": [logit_diff],
                }

                df = pd.concat(
                    [df, pd.DataFrame.from_dict(output_dict)],
                    ignore_index=True,
                )

                print("Saving csv")
                # Will overwrite this file after every evaluation
                df.to_csv(
                    os.path.join(config["results_dir"], "results.csv")
                )


if __name__ == "__main__":
    main()
