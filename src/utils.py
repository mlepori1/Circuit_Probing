import argparse
import sys

import torch
import torch.nn as nn
import yaml
from torch.nn import init
from torch.utils.data import DataLoader, random_split
from transformers import GPT2Config, GPT2LMHeadModel

from Datasets import AlgorithmicProbeDataset, SVAgrDataset, ReflexivesDataset, SyntacticNumberDataset
from NeuroSurgeon.NeuroSurgeon.Models.model_configs import CircuitConfig
from NeuroSurgeon.NeuroSurgeon.Probing.circuit_probe import CircuitProbe
from NeuroSurgeon.NeuroSurgeon.Probing.probe_configs import (
    CircuitProbeConfig,
    ResidualUpdateModelConfig,
)

def get_config():
    # Load config file from command line arg
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="where to load YAML configuration",
        metavar="FILE",
    )

    argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)

    if not hasattr(args, "config"):
        raise ValueError("Must include path to config file")
    else:
        with open(args.config, "r") as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)


def get_model(config):
    # Get model and handle initialization
    if config["random_init"] == False and config["layer_reinit"] == False:
        return GPT2LMHeadModel.from_pretrained(config["model_path"])
    elif config["random_init"] == True:
        if config["reinit_embeddings"] == True:
            return GPT2LMHeadModel(GPT2Config.from_pretrained(config["model_path"]))
        else:
            model = GPT2LMHeadModel.from_pretrained(config["model_path"])
            # Reinitialize everything but the embeddings
            model.transformer.h.apply(model._init_weights)
            model.transformer.ln_f.apply(model._init_weights)
            model.lm_head.apply(model._init_weights)
            return model
    elif config["layer_reinit"] == True:
        model = GPT2LMHeadModel.from_pretrained(config["model_path"])
        with torch.no_grad():
            if config["operation"] == "mlp":
                # Reinitialize target layer mlps by shuffling original weights
                orig_size = model.transformer.h[
                    config["target_layer"]
                ].mlp.c_fc.weight.shape
                flattened = model.transformer.h[
                    config["target_layer"]
                ].mlp.c_fc.weight.reshape(-1)
                perm = torch.randperm(len(flattened))
                flattened = flattened[perm]
                model.transformer.h[
                    config["target_layer"]
                ].mlp.c_fc.weight = nn.Parameter(flattened.reshape(orig_size))

                orig_size = model.transformer.h[
                    config["target_layer"]
                ].mlp.c_proj.weight.shape
                flattened = model.transformer.h[
                    config["target_layer"]
                ].mlp.c_proj.weight.reshape(-1)
                perm = torch.randperm(len(flattened))
                flattened = flattened[perm]
                model.transformer.h[
                    config["target_layer"]
                ].mlp.c_proj.weight = nn.Parameter(flattened.reshape(orig_size))
                return model

            elif config["operation"] == "attn":
                # Reinit attn linear layers by shuffling original weights
                orig_size = model.transformer.h[
                    config["target_layer"]
                ].attn.c_attn.weight.shape
                flattened = model.transformer.h[
                    config["target_layer"]
                ].attn.c_attn.weight.reshape(-1)
                perm = torch.randperm(len(flattened))
                flattened = flattened[perm]
                model.transformer.h[
                    config["target_layer"]
                ].attn.c_attn.weight = nn.Parameter(flattened.reshape(orig_size))

                orig_size = model.transformer.h[
                    config["target_layer"]
                ].attn.c_proj.weight.shape
                flattened = model.transformer.h[
                    config["target_layer"]
                ].attn.c_proj.weight.reshape(-1)
                perm = torch.randperm(len(flattened))
                flattened = flattened[perm]
                model.transformer.h[
                    config["target_layer"]
                ].attn.c_proj.weight = nn.Parameter(flattened.reshape(orig_size))

        return model
    else:
        raise ValueError(f'{config["model_type"]} is not supported')


def create_circuit_probe(config, model):
    # Create a circuit probe according to the config specifications
    if config["operation"] == "mlp":
        target_layers = [
            f'transformer.h.{config["target_layer"]}.mlp.c_fc',
            f'transformer.h.{config["target_layer"]}.mlp.c_proj',
        ]
    if config["operation"] == "attn":
        target_layers = [
            f'transformer.h.{config["target_layer"]}.attn.c_attn',
            f'transformer.h.{config["target_layer"]}.attn.c_proj',
        ]

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "neuron",
            "mask_bias": False,
            "mask_init_value": config["mask_init_value"],
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
        l0_lambda=config["l0_lambda"],
    )

    if config["operation"] == "mlp":
        resid_config = ResidualUpdateModelConfig(
            "gpt",
            target_layers=[config["target_layer"]],
            updates=True,
            stream=False,
            mlp=True,
            attn=False,
        )
        circuit_probe_config = CircuitProbeConfig(
            probe_vectors=f'mlp_update_{config["target_layer"]}',
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    elif config["operation"] == "attn":
        resid_config = ResidualUpdateModelConfig(
            "gpt",
            target_layers=[config["target_layer"]],
            updates=True,
            stream=False,
            mlp=False,
            attn=True,
        )
        circuit_probe_config = CircuitProbeConfig(
            probe_vectors=f'attn_update_{config["target_layer"]}',
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    else:
        raise ValueError("operation must be either mlp or attn")

    return CircuitProbe(circuit_probe_config, model)

## Dataset creation and splitting code
def create_datasets(config):
    train_dataset = AlgorithmicProbeDataset(
        config["train_data_path"], config["variable"], config["probe_index"]
    )
    test_dataset = AlgorithmicProbeDataset(
        config["test_data_path"], config["variable"], config["probe_index"]
    )

    trainloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    testloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )
    return trainloader, testloader


def create_sv_datasets(config, tokenizer):
    dataset = SVAgrDataset(config["data_path"], tokenizer, pad_max=15)

    remainder = len(dataset) - (config["train_size"] + config["test_size"])

    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    train_data, test_data, _ = random_split(
        dataset,
        [config["train_size"], config["test_size"], remainder],
        generator=generator,
    )
    trainloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    testloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )

    genset = SVAgrDataset(config["gen_path"], tokenizer, pad_max=20)

    remainder = len(genset) - config["test_size"]

    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    gen_data, _ = random_split(
        genset,
        [config["test_size"], remainder],
        generator=generator,
    )

    genloader = DataLoader(gen_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    return trainloader, testloader, genloader

def create_syntactic_number_datasets(config, tokenizer):
    # Train circuit probing to identify the syntactic number of the object noun
    dataset = SyntacticNumberDataset(config["data_path"], "object labels", tokenizer)
    # See if ablating this circuit ruins the ability of the network to identify the 
    # syntactic number of the subject noun
    lm_dataset = SyntacticNumberDataset(config["data_path"], "labels", tokenizer)

    remainder = len(dataset) - (config["train_size"] + config["test_size"])

    # Split circuit probe training and test data
    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    train_data, test_data, _ = random_split(
        dataset,
        [config["train_size"], config["test_size"], remainder],
        generator=generator,
    )

    # Split lm eval data, ensuring that the circuit probe test set is the same as the lm test set
    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    _, lm_test_data, _ = random_split(
        lm_dataset,
        [config["train_size"], config["test_size"], remainder],
        generator=generator,
    )

    # Assert that the same data partitions are used in lm eval and circuit probe test data
    assert test_data.indices == lm_test_data.indices

    trainloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    testloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )

    lm_testloader = DataLoader(lm_test_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    return trainloader, testloader, lm_testloader

def create_reflexive_datasets(config, tokenizer):
    male_dataset = ReflexivesDataset(config["data_path"], tokenizer, gender=0, pad_max=15)

    remainder = len(male_dataset) - (config["train_size"] + config["test_size"])

    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    male_train_data, male_test_data, _ = random_split(
        male_dataset,
        [config["train_size"], config["test_size"], remainder],
        generator=generator,
    )
    male_trainloader = DataLoader(
        male_train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    male_testloader = DataLoader(
        male_test_data, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )

    male_genset = ReflexivesDataset(config["gen_path"], tokenizer, gender=0, pad_max=20)

    remainder = len(male_genset) - config["test_size"]

    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    male_gen_data, _ = random_split(
        male_genset,
        [config["test_size"], remainder],
        generator=generator,
    )

    male_genloader = DataLoader(male_gen_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    female_dataset = ReflexivesDataset(config["data_path"], tokenizer, gender=1, pad_max=15)

    remainder = len(female_dataset) - (config["train_size"] + config["test_size"])

    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    _, female_test_data, _ = random_split(
        female_dataset,
        [config["train_size"], config["test_size"], remainder],
        generator=generator,
    )

    # Assert that the same data partitions are used in male and female data
    assert female_test_data.indices == male_test_data.indices

    female_testloader = DataLoader(
        female_test_data, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )

    female_genset = ReflexivesDataset(config["gen_path"], tokenizer, gender=1, pad_max=20)

    remainder = len(female_genset) - config["test_size"]

    torch.manual_seed(config["data_seed"])
    generator = torch.Generator().manual_seed(config["data_seed"])
    female_gen_data, _ = random_split(
        female_genset,
        [config["test_size"], remainder],
        generator=generator,
    )

    female_genloader = DataLoader(female_gen_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)

    return male_trainloader, male_testloader, male_genloader, female_testloader, female_genloader
