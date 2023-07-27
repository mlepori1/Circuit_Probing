from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    BertConfig,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Config,
    GPTNeoXForCausalLM,
    GPTNeoXConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    RobertaConfig,
    MPNetForMaskedLM,
    MPNetTokenizerFast,
    MPNetConfig,
    XLMRobertaForMaskedLM,
    XLMRobertaTokenizerFast,
    XLMRobertaConfig,
    ErnieForMaskedLM,
    AutoTokenizer,
    ErnieConfig,
    ElectraForMaskedLM,
    ElectraTokenizerFast,
    ElectraConfig,
    ConvBertForMaskedLM,
    ConvBertTokenizerFast,
    ConvBertConfig,
)

from ProbeDataset import ProbeDataset
from NeuroSurgeon.src.Models.model_configs import (
    CircuitConfig
    )
from NeuroSurgeon.src.Probing.circuit_probe import CircuitProbe
from NeuroSurgeon.src.Probing.subnetwork_probe import SubnetworkProbe
from NeuroSurgeon.src.Probing.probe_configs import CircuitProbeConfig, SubnetworkProbeConfig, ResidualUpdateModelConfig
import torch
from torch.nn import init
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
    if config["model_type"] == "bert" and config["random_init"] == False and config["layer_reinit"] == False:
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
    elif config["model_type"] == "bert" and config["layer_reinit"] == True:
        model = BertForMaskedLM.from_pretrained(
            config["model_path"]
        )
        if config["operation"] == "mlp":
            # Reinitialize just the target layer mlps
            init.normal_(
                model.bert.encoder.layer[config["target_layer"]].intermediate.dense.weight,
                mean=torch.mean(model.bert.encoder.layer[config["target_layer"]].intermediate.dense.weight).detach().item(), 
                std=torch.std(model.bert.encoder.layer[config["target_layer"]].intermediate.dense.weight).detach().item()
                )
            init.normal_(
                model.bert.encoder.layer[config["target_layer"]].output.dense.weight,
                mean=torch.mean(model.bert.encoder.layer[config["target_layer"]].output.dense.weight).detach().item(), 
                std=torch.std(model.bert.encoder.layer[config["target_layer"]].output.dense.weight).detach().item()  
            )
        elif config["operation"] == "attn":
            # Reinit attention linear layers
            init.normal_(
                model.bert.encoder.layer[config["target_layer"]].attention.self.query.weight,
                mean=torch.mean(model.bert.encoder.layer[config["target_layer"]].attention.self.query.weight).detach().item(),
                std=torch.std(model.bert.encoder.layer[config["target_layer"]].attention.self.query.weight).detach().item()
            )
            init.normal_(
                model.bert.encoder.layer[config["target_layer"]].attention.self.key.weight,
                mean=torch.mean(model.bert.encoder.layer[config["target_layer"]].attention.self.key.weight).detach().item(),
                std=torch.std(model.bert.encoder.layer[config["target_layer"]].attention.self.key.weight).detach().item()
            )
            init.normal_(
                model.bert.encoder.layer[config["target_layer"]].attention.self.value.weight,
                mean=torch.mean(model.bert.encoder.layer[config["target_layer"]].attention.self.value.weight).detach().item(),
                std=torch.std(model.bert.encoder.layer[config["target_layer"]].attention.self.value.weight).detach().item()
            )
            init.normal_(
                model.bert.encoder.layer[config["target_layer"]].attention.output.dense.weight,
                mean=torch.mean(model.bert.encoder.layer[config["target_layer"]].attention.output.dense.weight).detach().item(),
                std=torch.std(model.bert.encoder.layer[config["target_layer"]].attention.output.dense.weight).detach().item()
            )
        return model, BertTokenizerFast.from_pretrained(config["model_path"])

    elif config["model_type"] == "roberta" and config["random_init"] == False and config["layer_reinit"] == False:
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
    elif config["model_type"] == "roberta" and config["layer_reinit"] == True:
        model = RobertaForMaskedLM.from_pretrained(config["model_path"])
        if config["operation"] == "mlp":
            # Reinitialize target layer mlps
            init.normal_(
                model.roberta.encoder.layer[config["target_layer"]].intermediate.dense.weight,
                mean=torch.mean(model.roberta.encoder.layer[config["target_layer"]].intermediate.dense.weight).detach().item(), 
                std=torch.std(model.roberta.encoder.layer[config["target_layer"]].intermediate.dense.weight).detach().item()
                )
            init.normal_(
                model.roberta.encoder.layer[config["target_layer"]].output.dense.weight,
                mean=torch.mean(model.roberta.encoder.layer[config["target_layer"]].output.dense.weight).detach().item(), 
                std=torch.std(model.roberta.encoder.layer[config["target_layer"]].output.dense.weight).detach().item() 
            )
            return model, RobertaTokenizerFast.from_pretrained(
                config["model_path"], add_prefix_space=True
            )
        elif config["operation"] == "attn":
            # Reinit attn layer
            init.normal_(
                model.roberta.encoder.layer[config["target_layer"]].attention.self.query.weight,
                mean=torch.mean(model.roberta.encoder.layer[config["target_layer"]].attention.self.query.weight).detach().item(),
                std=torch.std(model.roberta.encoder.layer[config["target_layer"]].attention.self.query.weight).detach().item()
            )
            init.normal_(
                model.roberta.encoder.layer[config["target_layer"]].attention.self.key.weight,
                mean=torch.mean(model.roberta.encoder.layer[config["target_layer"]].attention.self.key.weight).detach().item(),
                std=torch.std(model.roberta.encoder.layer[config["target_layer"]].attention.self.key.weight).detach().item()
            )
            init.normal_(
                model.roberta.encoder.layer[config["target_layer"]].attention.self.value.weight,
                mean=torch.mean(model.roberta.encoder.layer[config["target_layer"]].attention.self.value.weight).detach().item(),
                std=torch.std(model.roberta.encoder.layer[config["target_layer"]].attention.self.value.weight).detach().item()
            )
            init.normal_(
                model.roberta.encoder.layer[config["target_layer"]].attention.output.dense.weight,
                mean=torch.mean(model.roberta.encoder.layer[config["target_layer"]].attention.output.dense.weight).detach().item(),
                std=torch.std(model.roberta.encoder.layer[config["target_layer"]].attention.output.dense.weight).detach().item()
            )
        return model, RobertaTokenizerFast.from_pretrained(
            config["model_path"], add_prefix_space=True
        )
    elif config["model_type"] == "gpt2" and config["random_init"] == False and config["layer_reinit"] == False:
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
    elif config["model_type"] == "gpt2" and config["layer_reinit"] == True:
            model = GPT2LMHeadModel.from_pretrained(config["model_path"])
            if config["operation"] == "mlp":
                # Reinitialize target layer mlps
                init.normal_(
                    model.transformer.h[config["target_layer"]].mlp.c_fc.weight,
                    mean=torch.mean(model.transformer.h[config["target_layer"]].mlp.c_fc.weight).detach().item(),
                    std=torch.std(model.transformer.h[config["target_layer"]].mlp.c_fc.weight).detach().item()
                )
                init.normal_(
                    model.transformer.h[config["target_layer"]].mlp.c_proj.weight,
                    mean=torch.mean(model.transformer.h[config["target_layer"]].mlp.c_proj.weight).detach().item(),
                    std=torch.std(model.transformer.h[config["target_layer"]].mlp.c_proj.weight).detach().item()
                )
                return model, GPT2TokenizerFast.from_pretrained(
                    config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
                )
            elif config["operation"] == "attn":
                # Reinit attn linear layers
                init.normal_(
                    model.transformer.h[config["target_layer"]].attn.c_attn.weight,
                    mean=torch.mean(model.transformer.h[config["target_layer"]].attn.c_attn.weight).detach().item(),
                    std=torch.std(model.transformer.h[config["target_layer"]].attn.c_attn.weight).detach().item()
                )
                init.normal_(
                    model.transformer.h[config["target_layer"]].attn.c_proj.weight,
                    mean=torch.mean(model.transformer.h[config["target_layer"]].attn.c_proj.weight).detach().item(),
                    std=torch.std(model.transformer.h[config["target_layer"]].attn.c_proj.weight).detach().item()
                )
            return model, GPT2TokenizerFast.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
            )
    elif config["model_type"] == "gpt_neox" and config["random_init"] == False and config["layer_reinit"] == False:
        conf = GPTNeoXConfig.from_pretrained(config["model_path"])
        conf.is_decoder = True
        tokenizer = AutoTokenizer.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
        )
        tokenizer.model_max_length=512

        return GPTNeoXForCausalLM.from_pretrained(
            config["model_path"], config=conf
        ), tokenizer
    elif config["model_type"] == "gpt_neox" and config["random_init"] == True:
        if config["reinit_embeddings"] == True:
            conf = GPTNeoXConfig.from_pretrained(config["model_path"])
            conf.is_decoder = True
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
            )
            tokenizer.model_max_length=512
            return GPTNeoXForCausalLM(
                conf
            ), tokenizer
        else:
            conf = GPTNeoXConfig.from_pretrained(config["model_path"])
            model = GPTNeoXForCausalLM.from_pretrained(config["model_path"], config=conf)
            # Reinitialize everything but the embeddings
            model.gpt_neox.layers.apply(model._init_weights)
            model.gpt_neox.final_layer_norm.apply(model._init_weights)
            model.embed_out.apply(model._init_weights)
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
            )
            tokenizer.model_max_length = 512
            return model, tokenizer
    elif config["model_type"] == "gpt_neox" and config["layer_reinit"] == True:
            conf = GPTNeoXConfig.from_pretrained(config["model_path"])
            model = GPTNeoXForCausalLM.from_pretrained(config["model_path"], config=conf)
            if config["operation"] == "mlp":
                # Reinitialize target layer mlps
                init.normal_(
                    model.gpt_neox.layers[config["target_layer"]].mlp.dense_h_to_4h.weight,
                    mean=torch.mean(model.gpt_neox.layers[config["target_layer"]].mlp.dense_h_to_4h.weight).detach().item(),
                    std=torch.std(model.gpt_neox.layers[config["target_layer"]].mlp.dense_h_to_4h.weight).detach().item()
                )
                init.normal_(
                    model.gpt_neox.layers[config["target_layer"]].mlp.dense_4h_to_h.weight,
                    mean=torch.mean(model.gpt_neox.layers[config["target_layer"]].mlp.dense_4h_to_h.weight).detach().item(),
                    std=torch.std(model.gpt_neox.layers[config["target_layer"]].mlp.dense_4h_to_h.weight).detach().item()
                )
            elif config["operation"] == "attn":
                init.normal_(
                    model.gpt_neox.layers[config["target_layer"]].attention.query_key_value.weight,
                    mean=torch.mean(model.gpt_neox.layers[config["target_layer"]].attention.query_key_value.weight).detach().item(),
                    std=torch.std(model.gpt_neox.layers[config["target_layer"]].attention.query_key_value.weight).detach().item()
                )
                init.normal_(
                    model.gpt_neox.layers[config["target_layer"]].attention.dense.weight,
                    mean=torch.mean(model.gpt_neox.layers[config["target_layer"]].attention.dense.weight).detach().item(),
                    std=torch.std(model.gpt_neox.layers[config["target_layer"]].attention.dense.weight).detach().item()
                )
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_path"], pad_token="<|endoftext|>", add_prefix_space=True
            )
            tokenizer.model_max_length = 512
            return model, tokenizer
    else:
        raise ValueError(f'{config["model_type"]} is not supported')


def create_circuit_probe(config, model, tokenizer):
    if config["model_type"] == "bert":
        if config["operation"] == "mlp":
            target_layers = [
                f'bert.encoder.layer.{config["target_layer"]}.intermediate.dense',
                f'bert.encoder.layer.{config["target_layer"]}.output.dense',
            ]
        if config["operation"] == "attn":
            target_layers = [
                f'bert.encoder.layer.{config["target_layer"]}.attention.self.query',
                f'bert.encoder.layer.{config["target_layer"]}.attention.self.key',
                f'bert.encoder.layer.{config["target_layer"]}.attention.self.value',
                f'bert.encoder.layer.{config["target_layer"]}.attention.output.dense',
            ]
    elif config["model_type"] == "roberta" or config["model_type"] == "xlm-roberta":
        if config["operation"] == "mlp":
            target_layers = [
                f'roberta.encoder.layer.{config["target_layer"]}.intermediate.dense',
                f'roberta.encoder.layer.{config["target_layer"]}.output.dense',
            ]
        if config["operation"] == "attn":
            target_layers = [
                f'roberta.encoder.layer.{config["target_layer"]}.attention.self.query',
                f'roberta.encoder.layer.{config["target_layer"]}.attention.self.key',
                f'roberta.encoder.layer.{config["target_layer"]}.attention.self.value',
                f'roberta.encoder.layer.{config["target_layer"]}.attention.output.dense',
            ]
    elif config["model_type"] == "gpt2":
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
    elif config["model_type"] == "gpt_neox":
        if config["operation"] == "mlp":
            target_layers = [
                f'gpt_neox.layers.{config["target_layer"]}.mlp.dense_h_to_4h',
                f'gpt_neox.layers.{config["target_layer"]}.mlp.dense_4h_to_h',
            ]
        if config["operation"] == "attn":
            target_layers = [
                f'gpt_neox.layers.{config["target_layer"]}.attention.query_key_value',
                f'gpt_neox.layers.{config["target_layer"]}.attention.dense',
            ]

    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": config["mask_unit"],
            "mask_bias": False,
            "mask_init_value": config["mask_init_value"],
        },
        target_layers=target_layers,
        freeze_base=True,
        add_l0=True,
        l0_lambda=config["l0_lambda"],
    )

    if config["model_type"] in ["bert",  "roberta"]:
        res_type = "bert"
    elif config["model_type"] == "gpt2":
        res_type = "gpt"
    elif config["model_type"] == "gpt_neox":
        res_type = "gpt_neox"

    if config["operation"] == "mlp":
        resid_config = ResidualUpdateModelConfig(
            res_type,
            target_layers=[config["target_layer"]],
            updates=config["updates"],
            residual=config["residual"],
            mlp=True,
            attn=False,
            circuit=True,
            base=False,
        )
        circuit_probe_config = CircuitProbeConfig(
            probe_activations=f'mlp_{config["target_layer"]}',
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    elif config["operation"] == "attn":
        resid_config = ResidualUpdateModelConfig(
            res_type,
            target_layers=[config["target_layer"]],
            updates=config["updates"],
            residual=config["residual"],
            mlp=False,
            attn=True,
            circuit=True,
            base=False,
        )
        circuit_probe_config = CircuitProbeConfig(
            probe_activations=f'attn_{config["target_layer"]}',
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    else:
        raise ValueError("operation must be either mlp or attn")

    return CircuitProbe(circuit_probe_config, model, tokenizer)


def create_subnetwork_probe(config, model, tokenizer):
   
   # Create a null circuit config
    circuit_config = CircuitConfig(
        mask_method="continuous_sparsification",
        mask_hparams={
            "ablation": "none",
            "mask_unit": "weight",
            "mask_bias": False,
            "mask_init_value": -1,
        },
        target_layers=[],
        freeze_base=True,
        add_l0=False,
        l0_lambda=0.0,
    )

    if config["model_type"] in ["bert",  "roberta"]:
        res_type = "bert"
    elif config["model_type"] == "gpt2":
        res_type = "gpt"
    elif config["model_type"] == "gpt_neox":
        res_type = "gpt_neox"

    if config["operation"] == "mlp":
        resid_config = ResidualUpdateModelConfig(
            res_type,
            target_layers=[config["target_layer"]],
            updates=config["updates"],
            residual=config["residual"],
            mlp=True,
            attn=False,
            circuit=True,
            base=False,
        )
        subnet_probe_config = SubnetworkProbeConfig(
            probe_activations=f'mlp_{config["target_layer"]}',
            intermediate_size=config["intermediate_size"],
            n_classes=config["n_classes"],
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    elif config["operation"] == "attn":
        resid_config = ResidualUpdateModelConfig(
            res_type,
            target_layers=[config["target_layer"]],
            updates=config["updates"],
            residual=config["residual"],
            mlp=False,
            attn=True,
            circuit=True,
            base=False,
        )
        subnet_probe_config = SubnetworkProbeConfig(
            probe_activations=f'attn_{config["target_layer"]}',
            intermediate_size=config["intermediate_size"],
            n_classes=config["n_classes"],
            circuit_config=circuit_config,
            resid_config=resid_config,
        )
    else:
        raise ValueError("operation must be either mlp or attn")

    return SubnetworkProbe(subnet_probe_config, model, tokenizer)

def create_datasets(config, tokenizer):
    dataset = ProbeDataset(
        config["train_data_path"], config["label"], tokenizer, seed=config["data_seed"]
    )
    remainder = len(dataset) - (
        config["train_size"] + config["dev_size"] + config["test_size"]
    )

    torch.manual_seed(config["data_seed"])
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
