import copy
import os
import shutil

import yaml

directories = [
    "Task_1/Shared",
    "Task_1/Unshared",
    "Task_2/Shared",
    "Task_2/Unshared",
]

for directory in directories:
    checkpoints = [5000, 10000, 20000, 30000, 40000, 50000]
    try:
        shutil.rmtree(os.path.join(directory, "Mask"))
    except:
        pass
    os.makedirs(os.path.join(directory, "Mask"), exist_ok=True)

    stream = open(os.path.join(directory, "mask_template.yaml"), "r")
    mask_template = yaml.load(stream, Loader=yaml.FullLoader)

    for ch in checkpoints:
        ch = ch

        temp = copy.deepcopy(mask_template)
        temp["model_path"] = temp["model_path"] + "_" + str(ch)
        temp["model_dir"] = os.path.join(temp["model_dir"], str(ch))
        temp["results_dir"] = os.path.join(temp["results_dir"], str(ch))
        with open(os.path.join(directory, "Mask", str(ch) + ".yaml"), "w") as outfile:
            yaml.dump(temp, outfile)
