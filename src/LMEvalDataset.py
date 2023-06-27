import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from file_readers import read_conll_format


class LMEvalDataset(Dataset):
    # A simple dataset class used to evaluate am ablated model's lm behavior
    def __init__(self, path, tokenizer, seed=0):
        random.seed(seed)

        if ".conll" in path:
            data = read_conll_format(path)

        random.shuffle(data)

        self.tokenizer = tokenizer
        self.x = []
        for d in data:
            self.x.append(d["text"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.tokenizer(
            [self.x[idx]],
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
        )
        sample = {
            "input_ids": x["input_ids"][0],
            "attention_mask": x["attention_mask"][0],
        }
        return sample


class MLMEvalDataset(Dataset):
    # A dataset class used for evaluating an ablated model's mlm behavior
    def __init__(self, path, tokenizer, seed=0):
        random.seed(seed)

        if ".conll" in path:
            data = read_conll_format(path)

        random.shuffle(data)

        self.tokenizer = tokenizer
        self.x = []
        for d in data:
            self.x.append(d["text"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Turns a single item into a batch of items, which contains copies of the data with masking at different indices
        x = self.tokenizer(
            [self.x[idx]],
            is_split_into_words=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        labels = x["input_ids"][0]

        # Don't mask special tokens

        # For every token, create a new row and mask if not a special token
        not_special = ~x["special_tokens_mask"].bool()
        not_special = not_special.repeat(not_special.size(1), 1)
        mask_idxs = (torch.eye(len(not_special)) * not_special).bool()

        # Mask out correct tokens
        ipt_ids = x["input_ids"].repeat(x["input_ids"].size(1), 1)
        ipt_ids[mask_idxs] = self.tokenizer.mask_token_id
        # If a row has no masked token, get rid of it
        # Should only get rid of rows corresponding to masking special tokens
        row_idxs = (mask_idxs.sum(dim=-1) != 0).bool()
        ipt_ids = ipt_ids[row_idxs]

        # Create one label set per row
        labels = labels.repeat(ipt_ids.size(0), 1)

        sample = {
            "input_ids": ipt_ids,
            "labels": labels,
        }
        return sample


if __name__ == "__main__":
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = LMEvalDataset("../data/UD/en-universal-train.conll", tok, seed=1)
    print(dataset[0])
    print(len(dataset))

    dataset = MLMEvalDataset("../data/UD/en-universal-train.conll", tok, seed=1)
    print(dataset[0])
    print(len(dataset))
