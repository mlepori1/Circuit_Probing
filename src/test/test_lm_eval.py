from ..lm_eval import get_lm_eval_data, lm_eval, masked_lm_eval
from ..utils import create_circuit_probe, get_model_and_tokenizer


def test_same_lm_when_not_ablating_bert():
    # Assert that LM evaluation numbers for BERT are not different if there are no masked parameters
    config = {
        "model_type": "bert",
        "random_init": False,
        "model_path": "bert-base-uncased",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": -1.0,  # Null subnet, ablating gives intact layer
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    lm_loader, _ = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    outs = lm_eval(config, model, tokenizer, lm_loader)

    assert outs["vanilla_acc"] == outs["ablated_acc"]
    assert outs["kl"] == 0


def test_same_lm_when_not_ablating_gpt():
    # Assert that lm evaluation numbers are not different for CLM if there are no masked parameters

    config = {
        "model_type": "gpt2",
        "random_init": False,
        "model_path": "gpt2",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": -1.0,  # Null subnet, ablating gives intact layer
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    lm_loader, _ = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    outs = lm_eval(config, model, tokenizer, lm_loader)

    assert outs["vanilla_acc"] == outs["ablated_acc"]
    assert outs["kl"] == 0


def test_same_mlm_when_not_ablating():
    # Assert that mlm evaluations are not different if there are no masked parameters

    config = {
        "model_type": "bert",
        "random_init": False,
        "model_path": "bert-base-uncased",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": -1.0,  # Null subnet, ablating gives intact layer
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    _, mlm_loader = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    outs = masked_lm_eval(config, model, tokenizer, mlm_loader)

    assert outs["mlm_vanilla_acc"] == outs["mlm_ablated_acc"]
    assert outs["mlm_kl"] == 0


def test_different_lm_when_ablating_bert():
    # Assert that LM evaluation numbers for BERT are different if there are masked parameters
    config = {
        "model_type": "bert",
        "random_init": False,
        "model_path": "bert-base-uncased",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": 1.0,  # Full subnet, ablating gives 0 layer
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    lm_loader, _ = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    outs = lm_eval(config, model, tokenizer, lm_loader)

    assert outs["vanilla_acc"] != outs["ablated_acc"]
    assert outs["kl"] > 0


def test_different_lm_when_ablating_gpt():
    # Assert that lm evaluation numbers are different for CLM if there are masked parameters
    config = {
        "model_type": "gpt2",
        "random_init": False,
        "model_path": "gpt2",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": 1.0,  # Full subnet, ablating gives 0 layer
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    lm_loader, _ = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    outs = lm_eval(config, model, tokenizer, lm_loader)

    assert outs["vanilla_acc"] != outs["ablated_acc"]
    assert outs["kl"] > 0


def test_different_mlm_when_ablating():
    # Assert that mlm evaluation numbers are different if there are masked parameters
    config = {
        "model_type": "bert",
        "random_init": False,
        "model_path": "bert-base-uncased",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": 1.0,  # Full subnet, ablating gives 0 layer
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    _, mlm_loader = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    outs = masked_lm_eval(config, model, tokenizer, mlm_loader)

    assert outs["mlm_vanilla_acc"] != outs["mlm_ablated_acc"]
    assert outs["mlm_kl"] > 0


def test_lm_better_than_mlm():
    # Assert that, in general, Bert models get higher accuracy on unmasked data than masked data
    config = {
        "model_type": "bert",
        "random_init": False,
        "model_path": "bert-base-uncased",
        "target_layer": 11,
        "mask_bias": False,
        "mask_init_value": -1.0,
        "l0_lambda": 0.0,
        "seed": 0,
        "test_size": 16,
        "batch_size": 4,
        "test_data_path": "test/sample.conllu",
    }
    model, tokenizer = get_model_and_tokenizer(config)
    probe = create_circuit_probe(config, model, tokenizer)
    probe.train(False)

    lm_loader, mlm_loader = get_lm_eval_data(config, tokenizer)

    model = probe.wrapped_model.model
    model.set_ablate_mode("zero_ablate")
    lm_outs = lm_eval(config, model, tokenizer, lm_loader)
    mlm_outs = masked_lm_eval(config, model, tokenizer, mlm_loader)

    assert lm_outs["vanilla_acc"] > mlm_outs["mlm_vanilla_acc"]
    assert lm_outs["ablated_acc"] > mlm_outs["mlm_ablated_acc"]
