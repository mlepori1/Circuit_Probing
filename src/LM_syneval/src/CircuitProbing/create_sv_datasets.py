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


outdir = "../../../../data/SV_Agr/"
os.makedirs(outdir, exist_ok=True)

sing_subject = set()
plur_subject = set()

datasets = [
    "obj_rel_across_anim.pickle",
    "obj_rel_across_inanim.pickle",
    "prep_anim.pickle",
    "prep_inanim.pickle",
]

for dataset in datasets:
    d = pkl.load(open(dataset, "rb"))
    sing_keys = []
    plur_keys = []
    for k in list(d.keys()):
        if k.startswith("sing"):
            sing_keys.append(k)
        if k.startswith("plur"):
            plur_keys.append(k)

    for sing_key in sing_keys:
        for pair in d[sing_key]:
            sing_subject.add(extract_pre_verb(pair[0]))
    for plur_key in plur_keys:
        for pair in d[plur_key]:
            plur_subject.add(extract_pre_verb(pair[0]))

sentences = list(sing_subject) + list(plur_subject)
labels = [0] * len(sing_subject) + [1] * len(plur_subject)
pd.DataFrame.from_dict({"sentence": sentences, "labels": labels}).to_csv(
    os.path.join(outdir, "SV_Agr.csv"), index=False
)

sing_subject_gen = set()
plur_subject_gen = set()

generalization_datasets = [    
    "obj_rel_across_anim_2.pickle",
    "obj_rel_across_inanim_2.pickle",
    "prep_anim_2.pickle",
    "prep_inanim_2.pickle",]

for dataset in generalization_datasets:
    d = pkl.load(open(dataset, "rb"))
    sing_keys = []
    plur_keys = []
    for k in list(d.keys()):
        if k.startswith("sing"):
            sing_keys.append(k)
        if k.startswith("plur"):
            plur_keys.append(k)

    for sing_key in sing_keys:
        for pair in d[sing_key]:
            sing_subject_gen.add(extract_pre_verb(pair[0]))
    for plur_key in plur_keys:
        for pair in d[plur_key]:
            plur_subject_gen.add(extract_pre_verb(pair[0]))


sentences = list(sing_subject_gen) + list(plur_subject_gen)
labels = [0] * len(sing_subject_gen) + [1] * len(plur_subject_gen)
gen_df = pd.DataFrame.from_dict({"sentence": sentences, "labels": labels})
gen_df = gen_df.sample(5000)
gen_df.to_csv(
    os.path.join(outdir, "SV_Agr_gen.csv"), index=False
)
