import copy
import os

import yaml


checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000]
os.makedirs(os.path.join("circuit_probing/a2"), exist_ok=True)

stream = open(os.path.join("circuit_probing_a2_template.yaml"), "r")
mask_template = yaml.load(stream, Loader=yaml.FullLoader)

for ch in checkpoints:
    temp = copy.deepcopy(mask_template)
    temp["model_path"] = temp["model_path"] + "_" + str(ch)
    temp["model_dir"] = os.path.join(temp["model_dir"], str(ch))
    temp["results_dir"] = os.path.join(temp["results_dir"], str(ch))
    with open(os.path.join("circuit_probing/a2", str(ch) + ".yaml"), "w") as outfile:
        yaml.dump(temp, outfile)

os.makedirs(os.path.join("circuit_probing/b2"), exist_ok=True)

stream = open(os.path.join("circuit_probing_b2_template.yaml"), "r")
mask_template = yaml.load(stream, Loader=yaml.FullLoader)

for ch in checkpoints:
    temp = copy.deepcopy(mask_template)
    temp["model_path"] = temp["model_path"] + "_" + str(ch)
    temp["model_dir"] = os.path.join(temp["model_dir"], str(ch))
    temp["results_dir"] = os.path.join(temp["results_dir"], str(ch))
    with open(os.path.join("circuit_probing/b2", str(ch) + ".yaml"), "w") as outfile:
        yaml.dump(temp, outfile)

