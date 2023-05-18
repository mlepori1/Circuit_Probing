import yaml
import sys
import os
import argparse
import pickle as pkl
from ProbeDataset import ProbeDataset
from transformers import BertConfig, BertForCircuitProbing, BertTokenizer, TrainingArguments

def get_config():
    # Load config file from command line arg
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)

    if not hasattr(args, "config"):
        raise ValueError("Must include path to config file")
    else:
        with open(args.config, 'r') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)

def load_dataset(config, tokenizer):
    build_path = lambda split : f"../data/ud-treebanks-v2.12/UD_English-EWT/en_ewt-ud-{split}.conllu"

    trainset = ProbeDataset(build_path("train"), config["task"], tokenizer)
    valset = ProbeDataset(build_path("dev"), config["task"], tokenizer)
    testset = ProbeDataset(build_path("test"), config["task"], tokenizer)

    return trainset, valset, testset

def create_model(tokenizer, config):

    if config["experiment_type"] == "circuit_probing":
        l0_start = config["l0_start"]
        l0_end = config["l0_end"]
        mask_init_value = config["mask_init_value"]
        ablate_mask = config["ablate_mask"]
        lamb = config["lambda"]
        probe_component = config["probe_component"]
        rm_loss = "soft_NN"

        bert_config = BertConfig.from_pretrained(config["pretrained_weights"])

        # Reset dropout probs to keep things in inference mode
        bert_config.hidden_dropout_prob = 0
        bert_config.attention_probs_dropout_prob = 0
        bert_config.classifier_dropout = 0

        bert_config.l0 = True
        bert_config.l0_start = config["l0_start"]
        bert_config.l0_end = config["l0_end"]
        bert_config.mask_init_value = config["mask_init_value"]
        bert_config.ablate_mask = config["ablate_mask"]
        bert_config.lamb = config["lambda"]
        bert_config.probe_component = config["probe_component"]
        bert_config.rm_loss = "soft_NN"

        if config["pretrained_weights"] is not None:
            model = BertForCircuitProbing.from_pretrained(config["pretrained_weights"], config=bert_config)
            model.set_tokenizer(tokenizer)
        else:
            model = BertForCircuitProbing(bert_config)
            model.set_tokenizer(tokenizer)
    
        if config["freeze_base"]:
            for layer in model.modules():
                if hasattr(layer, "weight"):
                    if layer.weight != None:
                        layer.weight.requires_grad = False
                if hasattr(layer, "bias"):
                    if layer.bias != None: 
                        layer.bias.requires_grad = False
        return model


def get_training_args(config, model_id):

    if config["experiment_type"] == "circuit_probing":
        # Ensure full batches for representation matching loss
        print("drop hanging batch")
        dataloader_drop_last = True
    else:
        dataloader_drop_last = False

    return TrainingArguments(
        output_dir=os.path.join(config["model_dir"], model_id),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        weight_decay=0.0,
        num_train_epochs=config["max_epochs"],
        lr_scheduler_type="linear",
        seed=config["seed"],
        data_seed=config["seed"],
        metric_for_best_model="eval_loss",
        load_best_model_at_end=config["load_best_model_at_end"],
        dataloader_drop_last=dataloader_drop_last
    )
