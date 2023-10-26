import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from transformers import GPT2Config
from utils import create_circuit_probe, get_model
from NeuroSurgeon.NeuroSurgeon.Visualization.visualizer import Visualizer, VisualizerConfig

# Analysis functions that load in pretrained circuitmodels and use NeuroSurgeon's visualizer to 
# plot the overlap between particular circuits
def create_agreement_overlap_graph():

    gpt2_cfg = GPT2Config.from_pretrained("gpt2")
    config = {
        "random_init": False,
        "layer_reinit": False,
        "model_path": "gpt2",
        "operation": "attn",
        "target_layer": 6,
        "mask_init_value": 0.0,
        "l0_lambda": 0.0
    }
    gpt2_model = get_model(config)
    sv_model = create_circuit_probe(config, gpt2_model)
    SV_WEIGHTS = "../Model/Probes/SV_Agr/small-save/de4ce015-5b7c-4b79-acd4-54d3c004dc94.pt"
    sv_model.load_state_dict(torch.load(SV_WEIGHTS))

    gpt2_model = get_model(config)
    reflexive_model = create_circuit_probe(config, gpt2_model)
    REFLEXIVE_WEIGHTS = "../Model/Probes/Reflexive_An/small-save/f00e4c6a-54d7-4277-a2ec-3c85ce18dbca.pt"
    reflexive_model.load_state_dict(torch.load(REFLEXIVE_WEIGHTS))

    viz_config = VisualizerConfig(
        model_list=[sv_model.wrapped_model.wrapped_model, reflexive_model.wrapped_model.wrapped_model], 
        model_labels=["S-V Agreement", "Reflexives"],
        subnetwork_colors=["cornflowerblue", "lightcoral"],
        intersect_color="plum",
        plot_full_network=False,
        plot_granularity="tensor",
        model_architecture="gpt2",
        hidden_size=gpt2_cfg.n_embd,
        num_heads=gpt2_cfg.n_head,
        figsize=(18, 11),
        format="pdf",
        outfile="analysis/Agreement/Small_Agreement_Subnetwork_Overlap_Tensor.pdf",
        title="GPT2-Small Agreement Circuit Overlap",
        legend=True,
        title_fontsize=30,
        label_fontsize=30
        )
    
    Visualizer(viz_config).plot()

    viz_config = VisualizerConfig(
        model_list=[sv_model.wrapped_model.wrapped_model, reflexive_model.wrapped_model.wrapped_model], 
        model_labels=["S-V Agreement", "Reflexives"],
        subnetwork_colors=["cornflowerblue", "lightcoral"],
        intersect_color="plum",
        plot_full_network=False,
        plot_granularity="block",
        model_architecture="gpt2",
        hidden_size=gpt2_cfg.n_embd,
        num_heads=gpt2_cfg.n_head,
        figsize=(18, 11),
        format="pdf",
        outfile="analysis/Agreement/Small_Agreement_Subnetwork_Overlap_Blocks.pdf",
        title="GPT2-Small Agreement Circuit Overlap - Blocks",
        legend=True,
        title_fontsize=30,
        label_fontsize=30
        )
    
    Visualizer(viz_config).plot()

def create_shared_nodes_overlap_graph():

    config = {
        "random_init": False,
        "layer_reinit": False,
        "model_path": "../Model/Algorithmic_Train/Shared_Nodes/model_LR_0.001_Seed_0_10000",
        "operation": "attn",
        "target_layer": 0,
        "mask_init_value": 0.0,
        "l0_lambda": 0.0
    }
    gpt2_model = get_model(config)
    task1_circuit_probe = create_circuit_probe(config, gpt2_model)
    TASK1_CIRCUIT_WEIGHT_PATH = "../Model/Probes/Shared_Nodes/Task_1/circuit_probing/free/c68eb9f7-7ca0-41a2-a158-5b49c4457f2c.pt"
    task1_circuit_probe.load_state_dict(torch.load(TASK1_CIRCUIT_WEIGHT_PATH))

    gpt2_model = get_model(config)
    task2_circuit_probe = create_circuit_probe(config, gpt2_model)
    TASK2_CIRCUIT_WEIGHT_PATH = "../Model/Probes/Shared_Nodes/Task_2/circuit_probing/free/c17a06f8-8fe5-4829-ae44-6c31f6acbfaa.pt"
    task2_circuit_probe.load_state_dict(torch.load(TASK2_CIRCUIT_WEIGHT_PATH))

    viz_config = VisualizerConfig(
        model_list=[task1_circuit_probe.wrapped_model.wrapped_model, task2_circuit_probe.wrapped_model.wrapped_model], 
        model_labels=["Task 1", "Task 2"],
        subnetwork_colors=["cornflowerblue", "lightcoral"],
        intersect_color="plum",
        plot_full_network=False,
        plot_granularity="block",
        model_architecture="gpt2",
        hidden_size=128,
        num_heads=4,
        figsize=(18, 11),
        format="pdf",
        outfile="analysis/Shared_Node/Task_Subnetwork_Overlap_Blocks.pdf",
        title="Free Variable Circuit Overlap: Blocks",
        legend=True,
        title_fontsize=30,
        label_fontsize=30
        )
    
    Visualizer(viz_config).plot()

    viz_config = VisualizerConfig(
        model_list=[task1_circuit_probe.wrapped_model.wrapped_model, task2_circuit_probe.wrapped_model.wrapped_model], 
        model_labels=["Task 1", "Task 2"],
        subnetwork_colors=["cornflowerblue", "lightcoral"],
        intersect_color="plum",
        plot_full_network=False,
        plot_granularity="tensor",
        model_architecture="gpt2",
        hidden_size=128,
        num_heads=4,
        figsize=(18, 11),
        format="pdf",
        outfile="analysis/Shared_Node/Task_Subnetwork_Overlap_Tensor.pdf",
        title="Free Variable Circuit Overlap",
        legend=True,
        title_fontsize=30,
        label_fontsize=30
        )
    
    Visualizer(viz_config).plot()


create_agreement_overlap_graph()
create_shared_nodes_overlap_graph()
