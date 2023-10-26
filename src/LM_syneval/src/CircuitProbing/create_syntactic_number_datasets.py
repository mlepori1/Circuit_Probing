import os
import pickle as pkl

import pandas as pd

# Need to extract prefix because the verb is not always in the same spot in the sentence
def extract_pre_verb(sentence):
    sentence = sentence.split(" ")
    if sentence[-2] == "is" or sentence[-2] == "are":
        sentence = " ".join(sentence[:-2])
    else:
        sentence = " ".join(sentence[:-1])
    return sentence


outdir = "../../../../data/Syntactic_Number/"
os.makedirs(outdir, exist_ok=True)

sing_subj_sing_obj = set()
sing_subj_plur_obj = set()
plur_subj_sing_obj = set()
plur_subj_plur_obj = set()

datasets = [
    "prep_anim.pickle",
    "prep_inanim.pickle",
]

# Annotate data by syntactic number of subject and object
for dataset in datasets:
    d = pkl.load(open(dataset, "rb"))
    sing_sing_keys = []
    sing_plur_keys = []
    plur_sing_keys = []
    plur_plur_keys = []
    for k in list(d.keys()):
        if k.startswith("sing"):
            if "plur" in k:
                sing_plur_keys.append(k)
            else:
                sing_sing_keys.append(k)

        if k.startswith("plur"):
            if "sing" in k:
                plur_sing_keys.append(k)
            else:
                plur_plur_keys.append(k)

    for k in sing_sing_keys:
        for pair in d[k]:
            sing_subj_sing_obj.add(extract_pre_verb(pair[0]))
    for k in sing_plur_keys:
        for pair in d[k]:
            sing_subj_plur_obj.add(extract_pre_verb(pair[0]))
    for k in plur_sing_keys:
        for pair in d[k]:
            plur_subj_sing_obj.add(extract_pre_verb(pair[0]))
    for k in plur_plur_keys:
        for pair in d[k]:
            plur_subj_plur_obj.add(extract_pre_verb(pair[0]))

sentences = list(sing_subj_sing_obj) + list(sing_subj_plur_obj) + list(plur_subj_sing_obj) + list(plur_subj_plur_obj)
subj_labels = [0] * len(sing_subj_sing_obj) + [0] * len(sing_subj_plur_obj)+ [1] * len(plur_subj_sing_obj) + [1] * len(plur_subj_plur_obj)
obj_labels = [0] * len(sing_subj_sing_obj) + [1] * len(sing_subj_plur_obj)+ [0] * len(plur_subj_sing_obj) + [1] * len(plur_subj_plur_obj)

pd.DataFrame.from_dict({"sentence": sentences, "labels": subj_labels, "object labels": obj_labels}).to_csv(
    os.path.join(outdir, "Syntactic_Number.csv"), index=False
)
