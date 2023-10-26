import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class AlgorithmicProbeDataset(Dataset):
    def __init__(self, file, variable, probe_index):
        """A Dataset to train circuit probes

        Args:
            file: Path to dataset
            variable: Which variable we're probing for
            probe_index: Which idx in the residual stream we're looking at
        """
        datafile = pd.read_csv(file)
        xs = []
        for data in datafile["data"]:
            x = data.split(" ")
            x = [int(number) for number in x]
            xs.append(torch.tensor(x).reshape(1, -1))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.tensor(datafile[variable])
        self.probe_target = probe_index

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        token_mask = torch.zeros(self.x[idx].shape)
        token_mask[self.probe_target] = 1
        sample = {
            "input_ids": self.x[idx],
            "labels": self.y[idx],
            "token_mask": token_mask.bool(),
        }
        return sample


class AlgorithmicTrainDataset(Dataset):
    def __init__(self, x, y):
        """A Dataset to train transformers"""
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            "input_ids": self.x[idx],
            "labels": self.y[idx],
        }
        return sample


class GifEvalDataset(Dataset):
    def __init__(self, file, variable, probe_index, num_classes=10, per_class=5):
        """A Dataset to create qualitative visualizations of the soft Nearest Neighbors training

        Args:
            file: Path to dataset
            variable: Which variable we're probing for
            probe_index: Which idx in the residual stream we're looking at
            num_classes: How many classes to include in the visualization. Defaults to 20.
            per_class: How many unique datapoints per class. Defaults to 10.
        """
        # Get Data from file
        datafile = pd.read_csv(file)
        xs = []
        for data in datafile["data"]:
            x = data.split(" ")
            x = [int(number) for number in x]
            xs.append(torch.tensor(x).reshape(1, -1))

        x = torch.cat(xs, dim=0)
        y = torch.tensor(datafile[variable])

        # Select classes to visualize and sample from them
        self.vis_x = []
        self.vis_y = []
        classes = np.random.choice(list(set(y)), size=num_classes, replace=False)
        for cl in classes:
            class_mask = y == cl
            class_x = x[class_mask]
            class_y = y[class_mask]
            sample = np.random.choice(
                list(range(len(class_x))), size=per_class, replace=False
            )
            self.vis_x.append(class_x[sample])
            self.vis_y.append(class_y[sample])

        self.probe_target = probe_index
        self.vis_x = np.concatenate(self.vis_x, axis=0)
        self.vis_y = np.concatenate(self.vis_y, axis=0)

    def __len__(self):
        return len(self.vis_x)

    def __getitem__(self, idx):
        token_mask = torch.zeros(self.vis_x[idx].shape)
        token_mask[self.probe_target] = 1
        sample = {
            "input_ids": self.vis_x[idx],
            "labels": self.vis_y[idx],
            "token_mask": token_mask.bool(),
        }
        return sample


class LMEvalDataset(Dataset):
    # A simple dataset class used to evaluate am ablated model's lm behavior
    def __init__(self, file):
        datafile = pd.read_csv(file)
        xs = []
        for data in datafile["data"]:
            x = data.split(" ")
            x = [int(number) for number in x]
            xs.append(torch.tensor(x).reshape(1, -1))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.tensor(datafile["labels"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {"input_ids": self.x[idx], "labels": self.y[idx]}
        return sample


class SVAgrDataset(Dataset):
    def __init__(self, file, tokenizer, pad_max=15):
        """A Dataset to train circuit probes

        Args:
            file: Path to dataset
            variable: Which variable we're probing for
        """
        datafile = pd.read_csv(file)
        xs = []
        probe_targets = []
        for sent in datafile["sentence"]:
            tokenized = tokenizer(sent)
            tok_len = len(tokenized["input_ids"])
            pad_len = pad_max - tok_len
            probe_targets.append(tok_len - 1)
            input_ids = tokenized["input_ids"] + ([tokenizer.eos_token_id] * pad_len)
            xs.append(torch.tensor(input_ids).reshape(1, -1))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.tensor(datafile["labels"])
        self.probe_target = probe_targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        token_mask = torch.zeros(self.x[idx].shape)
        token_mask[self.probe_target[idx]] = 1
        sample = {
            "input_ids": self.x[idx],
            "labels": self.y[idx],
            "token_mask": token_mask.bool(),
        }
        return sample

class SyntacticNumberDataset(Dataset):
    def __init__(self, file, label, tokenizer, pad_max=15):
        """A Dataset to train circuit probes for the syntactic number task

        Args:
            file: Path to dataset
        """
        datafile = pd.read_csv(file)
        xs = []
        probe_targets = []
        for sent in datafile["sentence"]:
            tokenized = tokenizer(sent)
            tok_len = len(tokenized["input_ids"])
            pad_len = pad_max - tok_len
            probe_targets.append(tok_len - 1)
            input_ids = tokenized["input_ids"] + ([tokenizer.eos_token_id] * pad_len)
            xs.append(torch.tensor(input_ids).reshape(1, -1))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.tensor(datafile[label])
        self.probe_target = probe_targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        token_mask = torch.zeros(self.x[idx].shape)
        token_mask[self.probe_target[idx]] = 1
        sample = {
            "input_ids": self.x[idx],
            "labels": self.y[idx],
            "token_mask": token_mask.bool(),
        }
        return sample


class ReflexivesDataset(Dataset):
    def __init__(self, file, tokenizer, pad_max=15, gender=-1):
        """A Dataset to train circuit probes

        Args:
            file: Path to dataset
            variable: Which variable we're probing for
        """
        datafile = pd.read_csv(file)
        if gender != -1:
            datafile = datafile[datafile["gender"] == gender]
        xs = []
        probe_targets = []
        for sent in datafile["sentence"]:
            tokenized = tokenizer(sent)
            tok_len = len(tokenized["input_ids"])
            pad_len = pad_max - tok_len
            probe_targets.append(tok_len - 2) # Probe the token right before the pronoun, pronoun is one token
            # Assert that the token after the target is one of the 3 possible pronouns [hardcoded gpt2 token values for himself, herself, themselves]
            assert tokenized["input_ids"][tok_len - 1] in [2241, 5223, 2405] 
            input_ids = tokenized["input_ids"] + ([tokenizer.eos_token_id] * pad_len)
            xs.append(torch.tensor(input_ids).reshape(1, -1))
        
        ungrammatical = []
        for sent in datafile["ungrammatical"]:
            tokenized = tokenizer(sent)
            tok_len = len(tokenized["input_ids"])
            pad_len = pad_max - tok_len
            # Assert that the token after the target is one of the 3 possible pronouns [hardcoded gpt2 token values for himself, herself, themselves]
            assert tokenized["input_ids"][tok_len - 1] in [2241, 5223, 2405] 
            input_ids = tokenized["input_ids"] + ([tokenizer.eos_token_id] * pad_len)
            ungrammatical.append(torch.tensor(input_ids).reshape(1, -1))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.tensor(datafile["labels"].values)
        self.ungrammatical = torch.cat(ungrammatical, dim=0)
        self.probe_target = probe_targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        token_mask = torch.zeros(self.x[idx].shape)
        token_mask[self.probe_target[idx]] = 1
        sample = {
            "input_ids": self.x[idx],
            "labels": self.y[idx],
            "token_mask": token_mask.bool(),
            "ungrammatical": self.ungrammatical[idx]
        }
        return sample

class ProbingDataset(Dataset):
    def __init__(self, file, probe_variable, device):
        """A Dataset to evaluate models using linear or nonlinear probing

        Args:
            file: Path to dataset
            probe_variables: Which intermediate variable to probe for

        """
        datafile = pd.read_csv(file)
        xs = []
        for data in datafile["data"]:
            x = data.split(" ")
            x = [int(number) for number in x]
            xs.append(torch.tensor(x).reshape(1, -1))

        self.x = torch.cat(xs, dim=0).to(device)
        self.y = torch.tensor(datafile[probe_variable].values).to(device)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            "input_ids": self.x[idx],
            "labels": self.y[idx],
        }
        return sample
     
class CounterfactualEmbeddingsDataset(Dataset):
    def __init__(self, file, counterfactual_label, counterfactual_variable, device):
        """A Dataset to evaluate models using Interchange Intervention 

        Args:
            file: Path to dataset
            counterfactual_label: Which counterfactual label we expect to recover after intervention
            counterfactual_variable: Which counterfactual variable to optimize against

        """
        datafile = pd.read_csv(file)
        xs = []
        for data in datafile["data"]:
            x = data.split(" ")
            x = [int(number) for number in x]
            xs.append(torch.tensor(x).reshape(1, -1))

        self.x = torch.cat(xs, dim=0).to(device)
        self.y = torch.tensor(datafile["labels"].values).to(device)
        self.cf_var = torch.tensor(datafile[counterfactual_variable].values).to(device)
        self.cf_label = torch.tensor(datafile[counterfactual_label].values).to(device)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            "input_ids": self.x[idx],
            "original_labels": self.y[idx],
            "counterfactual_variables": self.cf_var[idx],
            "counterfactual_labels": self.cf_label[idx],
        }
        return sample
    
class DASDataset(Dataset):
    def __init__(self, data, counterfactual_label, token_range, device="cuda"):
        self.token_range = tuple(token_range)

        input_ids = []
        for input in data["data"]:
            x = input.split(" ")
            x = [int(number) for number in x]
            input_ids.append(torch.tensor(x).reshape(1, -1))

        source_ids = []
        for input in data["counterfactual_data"]:
            cf = input.split(" ")
            cf = [int(number) for number in cf]
            source_ids.append(torch.tensor(cf).reshape(1, -1))

        self.source_input_ids = torch.cat(source_ids, dim=0).to(device)
        self.source_attention_mask = torch.ones(size=self.source_input_ids.shape).to(device)
        self.input_ids = torch.cat(input_ids, dim=0).to(device)
        self.attention_mask = torch.ones(size=self.input_ids.shape).to(device)

        self.labels = torch.ones(self.source_input_ids.shape) * -100
        self.labels[:, -1] = torch.tensor(data[counterfactual_label].values).reshape(-1)
        self.labels = self.labels.long().to(device)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        sample = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "source_input_ids": self.source_input_ids[idx],
            "source_attention_mask": self.source_attention_mask[idx],
            "labels": self.labels[idx],
            "token_range": self.token_range,
            "source_token_range": self.token_range,
            "intervention_ids": 0
        }
        return sample
    
if __name__ == "__main__":
    # data = AlgorithmicProbeDataset("../data/a_ab/train.csv", "var_1", 2)
    data = SVAgrDataset("../data/SV_Agr.csv", GPT2Tokenizer.from_pretrained("gpt2"))
    print(data[0])
    print(data[1])
