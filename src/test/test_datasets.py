from transformers import BertTokenizerFast
import torch
from ..ProbeDataset import ProbeDataset
from ..LMEvalDataset import MLMEvalDataset


def test_probe_dataset_conll_parsing():
    # Assert that Conll Parsing is correct
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = ProbeDataset("test_conll.conll", "tag", tok, seed=1)
    assert dataset[0]["labels"][0] == dataset.label2id.index("A")

    dataset = ProbeDataset("test_conll.conll", "pos", tok, seed=1)
    assert dataset[0]["labels"][0] == dataset.label2id.index("E")

    dataset = ProbeDataset("test_conll.conll", "dep", tok, seed=1)
    assert dataset[0]["labels"][0] == dataset.label2id.index("J")


def test_probe_dataset_token_masking():
    # Assert that token mask is correct, it ignores subwords and special tokens
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = ProbeDataset("test_conll.conll", "tag", tok, seed=1)
    ground_truth_tokens_mask = torch.Tensor(
        [False, True, True, True, True, False, False]
    )
    assert torch.all(
        dataset[0]["token_mask"][: ground_truth_tokens_mask.size(0)]
        == ground_truth_tokens_mask
    )
    assert torch.all(
        dataset[0]["token_mask"][ground_truth_tokens_mask.size(0) :] == False
    )  # assert all pad tokens are masked


def test_mlm_dataset_get_item():
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = MLMEvalDataset("test_conll.conll", tok, seed=1)
    assert (
        len(dataset[0]["input_ids"]) == 5
    )  # One input row for every token (even subwords)
    masking_token_mask = torch.tensor(
        [
            [False, True, False, False, False, False, False],
            [False, False, True, False, False, False, False],
            [False, False, False, True, False, False, False],
            [False, False, False, False, True, False, False],
            [False, False, False, False, False, True, False],
        ]
    )
    assert torch.all(
        (dataset[0]["input_ids"] == tok.mask_token_id) == masking_token_mask
    )  # Assert that only the expected tokens are replaced by mask_token_id
