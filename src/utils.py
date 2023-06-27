from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    BertConfig,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Config,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    RobertaConfig,
)

from ProbeDataset import ProbeDataset
from NeuroSurgeon.src.Models.model_configs import (
    CircuitConfig,
    ResidualUpdateModelConfig,
)
from NeuroSurgeon.src.Probing.circuit_probe import CircuitProbe
from NeuroSurgeon.src.Probing.probe_configs import CircuitProbeConfig
from torch.utils.data import random_split, DataLoader

import yaml
import sys
import argparse


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


def get_model_and_tokenizer(config):
    if config["model_type"] == "bert" and config["random_init"] == False:
        return BertForMaskedLM.from_pretrained(
            config["model_path"]
        ), BertTokenizerFast.from_pretrained(config["model_path"])
    elif config["model_type"] == "bert" and config["random_init"] == True:
        if config["reinit_embeddings"] == True:
            return BertForMaskedLM(
                BertConfig.from_pretrained(config["model_path"])
            ), BertTokenizerFast.from_pretrained(config["model_path"])
        else:
            model = BertForMaskedLM.from_pretrained(
                config["model_path"]
            )
            # Reinitialize everything but the embeddings
            model.bert.encoder.apply(model._init_weights)
            if model.bert.pooler is not None:
                model.bert.pooler.apply(model._init_weights)
            model.cls.apply(model._init_weights)
            return model, BertTokenizerFast.from_pretrained(config["model_path"])
    elif config["model_type"] == "roberta" and config["random_init"] == False:
        return RobertaForMaskedLM.from_pretrained(
            config["model_path"]
        ), RobertaTokenizerFast.from_pretrained(
            config["model_path"], add_prefix_space=True
        )
    elif config["model_type"] == "roberta" and config["random_init"] == True:
        if config["reinit_embeddings"] == True:
            return RobertaForMaskedLM(
                RobertaConfig.from_pretrained(config["model_path"])
            ), RobertaTokenizerFast.from_pretrained(
                config["model_path"], add_prefix_space=True
            )
        else:
            model = RobertaForMaskedLM.from_pretrained(config["model_path"])
            # Reinitialize everything but the embeddings
            model.roberta.encoder.apply(model._init_weights)
            if model.roberta.pooler is not None:
                model.roberta.pooler.apply(model._init_weights)
            model.lm_head.apply(model._init_weights)
            return model, RobertaTokenizerFast.from_pretrained(
                config["model_path"], add_prefix_space=True
            )
    elif config["model_type"] == "gpt2" and config["random_init"] == False:
        return GPT2LMHeadModel.from_pretrained(
            config["model_path"]
        ), GPT2TokenizerFast.from_pretrained(
            config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
        )
    elif config["model_type"] == "gpt2" and config["random_init"] == True:
        if config["reinit_embeddings"] == True:
            return GPT2LMHeadModel(
                GPT2Config.from_pretrained(config["model_path"])
            ), GPT2TokenizerFast.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
            )
        else:
            model = GPT2LMHeadModel.from_pretrained(config["model_path"])
            # Reinitialize everything but the embeddings
            model.transformer.h.apply(model._init_weights)
            model.transformer.ln_f.apply(model._init_weights)
            model.lm_head.apply(model._init_weights)
            return model, GPT2TokenizerFast.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
            )
    else:
        raise ValueError(f'{config["model_type"]} is not supported')


def create_circuit_probe(config, model, tokenizer):
    if config["model_type"] == "bert":
        target_layers = [
            f'bert.encoder.layer.{config["target_layer"]}.intermediate.dense',
            f'bert.encoder.layer.{config["target_layer"]}.output.dense',
        ]
    elif config["model_type"] == "roberta":
        target_layers = [
            f'roberta.encoder.layer.{config["target_layer"]}.intermediate.dense',
            f'roberta.encoder.layer.{config["target_layer"]}.output.dense',
        ]
    elif config["model_type"] == "gpt2":
        target_layers = [
            f'transformer.h.{config["target_layer"]}.mlp.c_fc',
            f'transformer.h.{config["target_layer"]}.mlp.c_proj',
        ]

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_bias": config["mask_bias"],
            "mask_init_value": config["mask_init_value"],
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
        l0_lambda=config["l0_lambda"],
    )

    if config["model_type"] == "bert":
        res_type = "bert"
    elif config["model_type"] == "roberta":
        res_type = "bert"
    elif config["model_type"] == "gpt2":
        res_type = "gpt"

    resid_config = ResidualUpdateModelConfig(
        res_type,
        target_layers=[config["target_layer"]],
        mlp=True,
        attn=False,
        circuit=True,
        base=False,
    )

    circuit_probe_config = CircuitProbeConfig(
        probe_updates=f'mlp_{config["target_layer"]}',
        circuit_config=circuit_config,
        resid_config=resid_config,
    )

    return CircuitProbe(circuit_probe_config, model, tokenizer)


def create_datasets(config, tokenizer):
    dataset = ProbeDataset(
        config["train_data_path"], config["label"], tokenizer, seed=config["seed"]
    )
    remainder = len(dataset) - (
        config["train_size"] + config["dev_size"] + config["test_size"]
    )
    train_data, dev_data, test_data, _ = random_split(
        dataset,
        [config["train_size"], config["dev_size"], config["test_size"], remainder],
    )
    trainloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    devloader = DataLoader(
        dev_data, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )
    testloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )
    return trainloader, devloader, testloader
