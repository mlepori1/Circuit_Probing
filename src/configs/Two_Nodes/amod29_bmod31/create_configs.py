import copy
import os

import yaml

directories = ["A", "B"]

for directory in directories:
    checkpoints = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
    os.makedirs(os.path.join(directory, "Full"), exist_ok=True)
    os.makedirs(os.path.join(directory, "Mask"), exist_ok=True)

    stream = open(os.path.join(directory, "full_template.yaml"), "r")
    full_template = yaml.load(stream, Loader=yaml.FullLoader)

    stream = open(os.path.join(directory, "mask_template.yaml"), "r")
    mask_template = yaml.load(stream, Loader=yaml.FullLoader)

    for ch in checkpoints:
        ch = ch 
        temp = copy.deepcopy(full_template)
        temp["model_path"] = temp["model_path"] + "_" + str(ch)
        temp["model_dir"] = os.path.join(temp["model_dir"], str(ch))
        temp["results_dir"] = os.path.join(temp["results_dir"], str(ch))
        with open(os.path.join(directory, "Full", str(ch) + ".yaml"), "w") as outfile:
            yaml.dump(temp, outfile)

        temp = copy.deepcopy(mask_template)
        temp["model_path"] = temp["model_path"] + "_" + str(ch)
        temp["model_dir"] = os.path.join(temp["model_dir"], str(ch))
        temp["results_dir"] = os.path.join(temp["results_dir"], str(ch))
        with open(os.path.join(directory, "Mask", str(ch) + ".yaml"), "w") as outfile:
            yaml.dump(temp, outfile)
