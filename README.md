# Circuit Probing
Codebase for **Uncovering Causal Variables in Transformers Using Circuit Probing**

## Citations
This codebase relies on NeuroSurgeon (https://github.com/mlepori1/NeuroSurgeon), transformer-lens (https://github.com/neelnanda-io/TransformerLens), align_transformers (https://github.com/frankaging/align-transformers), and the codebase associated with Marvin & Linzen 2018 (https://github.com/BeckyMarvin/LM_syneval/tree/master).

## Reproducing our results

### Experiment 1
To reproduce Experiment 1 (Deciphering Neural Network Algorithms), first train a 1-layer GPT2 model by running `python train_algorithmic_models.py --config configs/Train_Algorithmic_Models/Disambiguation/a2_minus_b2.yaml`. This will create a model in the `Models` folder. Now, you can reproduce all of our analyses of this model using the following commands. Each analysis takes in a configuration file, which defines the specific variable that we'd like to probe for. Look inside these configuration files for the path containing results.

- Circuit Probing `python circuit_probing.py --configs/Disambiguation/circuit_probing/**INSERT FILE**`
- Boundless DAS `python DAS.py --configs/Disambiguation/DAS/**INSERT FILE**`
- Linear Probing and Counterfactual Embeddings `python probing.py --configs/Disambiguation/linear_probing/**INSERT FILE**`
- Nonlinear Probing and Counterfactual Embeddings `python probing.py --configs/Disambiguation/nonlinear_probing/**INSERT FILE**`
- Transfer Learning `python train_algorithmic_models.py --config/Disambiguation/transfer/**INSERT FILE**`

### Experiment 2
To reproduce Experiment 2 (Modularity of Intermediate Variables), first train a 1-layer GPT2 model by running `python train_algorithmic_models.py --config configs/Train_Algorithmic_Models/Shared_Nodes/shared_nodes.yaml`. This will create a model in the `Models` folder. Now, you can reproduce all of our analyses of this model using the following commands. Each analysis takes in a configuration file, which defines the specific variable that we'd like to probe for. All methods are trained on data within _one_ of the tasks in the dataset, so the configuration files are split into tasks. Look inside these configuration files for the path containing results.

- Circuit Probing `python circuit_probing.py --configs/Shared_Nodes/**INSERT TASK/circuit_probing/**INSERT FILE**`
- Boundless DAS `python DAS.py --configs/Shared_Nodes/**INSERT TASK/DAS/**INSERT FILE**`
- Linear Probing and Counterfactual Embeddings `python probing.py --configs/Shared_Nodes/**INSERT TASK/linear_probing/**INSERT FILE**`
- Nonlinear Probing and Counterfactual Embeddings `python probing.py --configs/Shared_Nodes/**INSERT TASK/nonlinear_probing/**INSERT FILE**`

### Experiment 3
To reproduce Experiment 3 (Circuit Probing as a Progress Measure), first train a 1-layer GPT2 model by running `python train_algorithmic_models.py --config configs/Train_Algorithmic_Models/Grokking/a2_b.yaml`. This will create a model in the `Models` folder. Now, you can reproduce all of our analyses of this model using the following commands. Each analysis takes in a configuration file, which defines the specific variable that we'd like to probe for. These analyses take place at multiple checkpoints during training, so you must specify both the variable to probe for, as well as the checkpoint, within the path. Look inside these configuration files for the path containing results.

- Circuit Probing `python circuit_probing.py --configs/Grokking/circuit_probing/circuit_probing/**INSERT VARIABLE**/**INSERT FILE**`
- Linear Probing `python probing.py --configs/Grokking/linear_probing/**INSERT VARIABLE**/**INSERT FILE**`
- Nonlinear Probing  `python probing.py --configs/Grokking/nonlinear_probing/**INSERT VARIABLE**/**INSERT FILE**`
  
### Experiment 4
To reproduce our subject-verb agreement and reflexive anaphora results, simply run 
`python circuit_probing.py` and point the `--config` argument to a file in `configs/SV_Agreement` or `configs/Reflexives`.

