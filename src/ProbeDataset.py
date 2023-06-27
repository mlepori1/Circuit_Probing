import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast
from file_readers import read_conll_format


class ProbeDataset(Dataset):
    def __init__(self, path, label, tokenizer, seed=0):
        random.seed(seed)

        if ".conll" in path:
            data = read_conll_format(path)
            self.label2id = self.create_label2id(data, label)

        random.shuffle(data)

        self.tokenizer = tokenizer
        self.x = []
        self.y = []
        for d in data:
            self.x.append(d["text"])

            # Pad y values for batching
            y = [self.label2id.index(lab) for lab in d["labels"][label]]
            remainder = self.tokenizer.model_max_length - len(y)
            y += [-1] * remainder
            y = torch.tensor(y)
            self.y.append(y)

    def create_label2id(self, data, label):
        label2id = set()
        for d in data:
            label2id.update(d["labels"][label])
        label2id = list(label2id)
        label2id.sort()
        return label2id

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.tokenizer(
            [self.x[idx]],
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
            padding="max_length",
        )

        # Compute token mask for probing
        offset_map = x["offset_mapping"]
        not_subword = offset_map[:, :, 0] == 0

        # Compute mask to get rid of special tokens
        not_special = ~x["special_tokens_mask"].bool()

        # Elementwise multiplication of masks to get first token for each word
        token_mask = not_subword * not_special
        token_mask = token_mask.bool()

        # BPE Tokenization sometimes acts funky on pretokenized data, giving tokens that are just comprised of one unicode character (i.e. just the prepended space token)
        # Ignore these as well
        if type(self.tokenizer) == RobertaTokenizerFast or type(self.tokenizer) == GPT2TokenizerFast:
            not_prepended_space = torch.tensor([len(el) > 1 for el in self.tokenizer.convert_ids_to_tokens(x["input_ids"][0])])
            token_mask = (token_mask * not_prepended_space).bool()

        y = self.y[idx]

        sample = {
            "input_ids": x["input_ids"][0],
            "attention_mask": x["attention_mask"][0],
            "token_mask": token_mask[0],
            "labels": y,
        }

        return sample


if __name__ == "__main__":
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = ProbeDataset(
        "../data/ud-treebanks-conll2017/UD_English/en-ud-train.conllu",
        "tag",
        tok,
        seed=0,
    )
    # print(dataset[0])
    print(len(dataset.label2id))
    print(len(dataset))
