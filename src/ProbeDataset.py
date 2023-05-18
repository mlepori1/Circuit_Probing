from torch.utils.data import Dataset
import torch
import random
from data_utils import read_conll_format, read_onto_notes_format

class ProbeDataset(Dataset):

    def __init__(self, path, label, tokenizer, seed=0):
        random.seed(seed)
        
        if ".conllu" in path:
            data = read_conll_format(path)
            label2id = self.create_label2id(data, label)

        random.shuffle(data)
                      
        self.tokenizer = tokenizer
        self.x = []
        self.y = []
        for d in data:
            self.x.append(" ".join(d["text"]))
            self.y.append([label2id.index(lab) for lab in d["labels"][label]])

    def create_label2id(self, data, label):
        label2id = set()
        for d in data:
            label2id.update(d["labels"][label])
        return list(label2id)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.tokenizer([self.x[idx]])
        y = self.y[idx]
        sample = {
            'input_ids': x["input_ids"][0], 
            'attention_mask': x["attention_mask"][0], 
            'token_type_ids': x["token_type_ids"][0], 
            'label': y}
        return sample