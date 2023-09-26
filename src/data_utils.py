import os

import einops
import pandas as pd
import torch


def generate_variable_function_results(variable_function, dataset, a_idx, b_idx, mod):
        # Pass in the right tokens as function args
            # If modulo is set for a function, don't pass it in (i.e. for mod N functions)
        if (
            hasattr(variable_function, "keywords")
            and variable_function.keywords["p"] is not None
        ):
            intermediate_output = variable_function(dataset[:,a_idx], dataset[:, b_idx])
        else:
            intermediate_output = variable_function(
                dataset[:, a_idx], dataset[:, b_idx], mod
            )

        return intermediate_output

def join_data(data, counterfactual_data, labels, intermediate_variables, counterfactual_intermediate_variables, auxiliary_variables, counterfactual_auxiliary_variables, indices):
    data = data[indices]
    counterfactual_data = counterfactual_data[indices]
    labels = labels[indices]
    intermediate_variables = [
        int_vars[indices] for int_vars in intermediate_variables
    ]
    counterfactual_variables = [
        int_vars[indices] for int_vars in counterfactual_intermediate_variables
    ]

    auxiliary_variables = [
        aux_vars[indices] for aux_vars in auxiliary_variables
    ]
    counterfactual_auxiliary_variables = [
        aux_vars[indices] for aux_vars in counterfactual_auxiliary_variables
    ]

    # Create Data Dict
    str_data = [
        " ".join([str(el.item()) for el in data]) for data in data
    ]
    str_counterfactual_data = [
        " ".join([str(el.item()) for el in data]) for data in counterfactual_data
    ]
    dict = {
        "data": str_data,
        "labels": labels,
        "counterfactual_data": str_counterfactual_data
    }

    for i in range(len(intermediate_variables)):
        dict["var_" + str(i)] = intermediate_variables[i]
        dict["counter_var_" + str(i)] = counterfactual_variables[i]

    for i in range(len(auxiliary_variables)):
        dict["aux_" + str(i)] = auxiliary_variables[i]
        dict["counter_aux_" + str(i)] = counterfactual_auxiliary_variables[i]

    df = pd.DataFrame.from_dict(dict)

    return df, data, labels

def add_cf_labels(df, label_functions, mod):
    """_summary_

    :param df: dataframe with all intermediate variables, auxiliary variables, and counterfactual variables
    :type df: pd.DataFrame
    :param counterfactual_label_functions: A list of lists of the form ["var_1", "var_2", "operation"] which determines the counterfactual label
        to be included in the dataframe
    :type counterfactual_label_functions: List[List[Str]]
    """
    for i in range(len(label_functions)):
        label_function = label_functions[i]
        var1 = df[label_function[0]]
        var2 = df[label_function[1]]
        if label_function[2] == "+":
            label = (var1 + var2) % mod
        elif label_function[2] == "*":
            label = (var1 * var2) % mod
        df["CF_Label_" + str(i)] = label
        df["CF_FN_" + str(i)] = " ".join(label_function)
   
    return df
        
def generate_data(
    intermediate_variable_functions,
    mod=113,
    train_frac=0.25,
    device="cuda",
    data_path="/data/",
    task_id=None,
    insert_random_tokens=None,
    label_function=["var_0", "var_1", "+"],
    auxiliary_variable_functions=[],
    counterfactual_label_functions=[],
):
    os.makedirs(data_path, exist_ok=True)
    # Input format is A B =
    # Code snippet inspired by Neel Nanda's Grokking Demo ipynb
    a_vector = einops.repeat(torch.arange(mod), "i -> (i j)", j=mod)
    b_vector = einops.repeat(torch.arange(mod), "j -> (i j)", i=mod)
    equals_vector = einops.repeat(torch.tensor(mod), " -> (i j)", i=mod, j=mod)
    if task_id is not None:
        task_vector = einops.repeat(torch.tensor(task_id), " -> (i j)", i=mod, j=mod)
        dataset = torch.stack([task_vector, a_vector, b_vector, equals_vector], dim=1)
    else:
        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)

    # Add random vector, if that is necessary (i.e. during multitask training)
    if insert_random_tokens is not None:
        random_vector = einops.repeat(torch.randperm(mod), "j -> (i j)", i=mod)
        dataset = torch.cat([dataset[:, :insert_random_tokens], random_vector.reshape(-1, 1), dataset[:, insert_random_tokens:]], dim=1)

    indices = torch.randperm(mod * mod)

    # Resample counterfactuals until all counterfactuals are different from the samples
    while not torch.all(torch.arange(len(dataset)) != indices):
        indices = torch.randperm(mod * mod)

    counterfactual_dataset = dataset[indices]

    # Functions always take the form F1(A, B, mod) + F2(A, B, mod) + ... % mod
    intermediate_variables = []
    intermediate_counterfactual_variables = []
    for fn_tuple in intermediate_variable_functions:
        variable_function = fn_tuple[0]
        a_idx = fn_tuple[1]
        b_idx = fn_tuple[2]
        intermediate_variables.append(generate_variable_function_results(variable_function, dataset, a_idx, b_idx, mod))
        intermediate_counterfactual_variables.append(generate_variable_function_results(variable_function, counterfactual_dataset, a_idx, b_idx, mod))

    auxiliary_variables = []
    auxiliary_counterfactual_variables = []
    for fn_tuple in auxiliary_variable_functions:
        variable_function = fn_tuple[0]
        a_idx = fn_tuple[1]
        b_idx = fn_tuple[2]
        auxiliary_variables.append(generate_variable_function_results(variable_function, dataset, a_idx, b_idx, mod))
        auxiliary_counterfactual_variables.append(generate_variable_function_results(variable_function, counterfactual_dataset, a_idx, b_idx, mod))

    intermediate_variables = torch.stack(intermediate_variables)
    intermediate_counterfactual_variables = torch.stack(intermediate_counterfactual_variables)
    if auxiliary_variables != []:
        auxiliary_variables = torch.stack(auxiliary_variables)
        auxiliary_counterfactual_variables = torch.stack(auxiliary_counterfactual_variables)

    # Labels are always summations over intermediate variables, mod P
    labels = intermediate_variables.sum(0) % mod

    indices = torch.randperm(mod * mod)
    cutoff = int(mod * mod * train_frac)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]


    train_df, train_data, train_labels = join_data(dataset, counterfactual_dataset, labels, intermediate_variables, intermediate_counterfactual_variables, auxiliary_variables, auxiliary_counterfactual_variables, train_indices)
    test_df, test_data, test_labels = join_data(dataset, counterfactual_dataset, labels, intermediate_variables, intermediate_counterfactual_variables, auxiliary_variables, auxiliary_counterfactual_variables, test_indices)

    train_df = add_cf_labels(train_df, counterfactual_label_functions, mod)
    test_df = add_cf_labels(test_df, counterfactual_label_functions, mod)


    train_df.to_csv(os.path.join(data_path, "train.csv"))
    test_df.to_csv(os.path.join(data_path, "test.csv"))

    return (
        train_data.to(device),
        train_labels.to(device),
        test_data.to(device),
        test_labels.to(device),
    )


def generate_multitask_data(
    task_1_functions,
    task_2_functions,
    mod=113,
    train_frac=0.25,
    device="cuda",
    data_path="/data/",
    task_1_aux_functions=[],
    task_2_aux_functions=[],
    task_1_counterfactual_label_functions=[],
    task_2_counterfactual_label_functions=[]
):
    train_x_1, train_y_1, test_x_1, test_y_1 = generate_data(
        task_1_functions,
        mod=mod,
        insert_random_tokens=2,
        train_frac=train_frac,
        device=device,
        data_path=os.path.join(data_path, "Task_1"),
        task_id=mod + 1,
        auxiliary_variable_functions=task_1_aux_functions,
        counterfactual_label_functions=task_1_counterfactual_label_functions,
    )
    train_x_2, train_y_2, test_x_2, test_y_2 = generate_data(
        task_2_functions,
        mod=mod,
        insert_random_tokens=3,
        train_frac=train_frac,
        device=device,
        data_path=os.path.join(data_path, "Task_2"),
        task_id=mod + 2,
        auxiliary_variable_functions=task_2_aux_functions,
        counterfactual_label_functions=task_2_counterfactual_label_functions,
    )

    train_data = torch.cat([train_x_1, train_x_2])
    train_labels = torch.cat([train_y_1, train_y_2])
    test_data = torch.cat([test_x_1, test_x_2])
    test_labels = torch.cat([test_y_1, test_y_2])

    return (
        train_data.to(device),
        train_labels.to(device),
        test_data.to(device),
        test_labels.to(device),
    )


def a_identity(a, b, p):
    return a % p


def a2(a, b, p):
    return a**2 % p


def a4(a, b, p):
    return a**4 % p


def a_mod(a, b, p):
    return a % p


def b_identity(a, b, p):
    return b % p


def b2(a, b, p):
    return b**2 % p


def minus_b2(a, b, p):
    return (-(b**2)) % p


def b4(a, b, p):
    return b**4 % p


def b_mod(a, b, p):
    return b % p


def ab(a, b, p):
    return (a * b) % p


def a_plus_b(a, b, p):
    return (a + b) % p


def a_minus_b(a, b, p):
    return (a - b) % p


def a_plus_b_no_prime(a, b, p):
    return a + b


def a_minus_b_no_prime(a, b, p):
    return a - b