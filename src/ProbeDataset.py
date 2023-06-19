from torch.utils.data import Dataset
import torch
import random
from amnesic_probing.amnesic_probing.encoders import read_conll_format
from transformers import BertTokenizerFast

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
            self.y.append([self.label2id.index(lab) for lab in d["labels"][label]])

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
        print(self.x[idx])
        x = self.tokenizer([self.x[idx]], is_split_into_words=True, return_tensors="pt")
        y = self.y[idx]
        sample = {
            'input_ids': x["input_ids"][0], 
            'attention_mask': x["attention_mask"][0], 
            'token_type_ids': x["token_type_ids"][0], 
            'label': y}
        return sample


if __name__=="__main__":
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = ProbeDataset("../data/UD/en-universal-train.conll", "tag", tok, seed=1)
    print(dataset[0])
    print(len(dataset.label2id))
    print(len(dataset))