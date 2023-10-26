import os
import pickle as pkl

import pandas as pd

outdir = "../../../../data/Reflexive_An/"
os.makedirs(outdir, exist_ok=True)

sing_subject = []
sing_ungrammatical = []
plur_subject = []
plur_ungrammatical =[]
gender = []

datasets = [
    "reflexive_sent_comp.pickle",
    "reflexives_across.pickle",
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
            sing_subject.append(pair[0])
            sing_ungrammatical.append(pair[1])
            if "herself" in pair[0] or "herself" in pair[1]:
                gender.append(1)
            else:
                gender.append(0)
    for plur_key in plur_keys:
        for pair in d[plur_key]:
            plur_subject.append(pair[0])
            plur_ungrammatical.append(pair[1])
            if "herself" in pair[0] or "herself" in pair[1]:
                gender.append(1)
            else:
                gender.append(0)

sentences = list(sing_subject) + list(plur_subject)
ungrammatical_sentences = sing_ungrammatical + plur_ungrammatical
labels = [0] * len(sing_subject) + [1] * len(plur_subject)
pd.DataFrame.from_dict({"sentence": sentences, "labels": labels, "ungrammatical": ungrammatical_sentences, "gender": gender}).to_csv(
    os.path.join(outdir, "Reflexive_An.csv"), index=False
)

sing_subject_gen = []
sing_ungrammatical_gen = []
plur_subject_gen = []
plur_ungrammatical_gen = []
gender_gen = []

generalization_datasets = [
    "reflexive_sent_comp_2.pickle",
    "reflexives_across_2.pickle"]

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
            sing_subject_gen.append(pair[0])
            sing_ungrammatical_gen.append(pair[1])
            if "herself" in pair[0] or "herself" in pair[1]:
                gender_gen.append(1)
            else:
                gender_gen.append(0)
    for plur_key in plur_keys:
        for pair in d[plur_key]:
            plur_subject_gen.append(pair[0])
            plur_ungrammatical_gen.append(pair[1])
            if "herself" in pair[0] or "herself" in pair[1]:
                gender_gen.append(1)
            else:
                gender_gen.append(0)

sentences = list(sing_subject_gen) + list(plur_subject_gen)
ungrammatical_sentences = sing_ungrammatical_gen + plur_ungrammatical_gen
labels = [0] * len(sing_subject_gen) + [1] * len(plur_subject_gen)
gen_df = pd.DataFrame.from_dict({"sentence": sentences, "labels": labels, "ungrammatical": ungrammatical_sentences, "gender": gender_gen})
gen_df = gen_df.sample(5000)
gen_df.to_csv(
    os.path.join(outdir, "Reflexive_An_gen.csv"), index=False
)
