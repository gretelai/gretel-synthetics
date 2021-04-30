from pathlib import Path
from copy import deepcopy

import pytest


from gretel_synthetics.config import BaseConfig
import gretel_synthetics.tokenizers as tok


class SimpleConfig(BaseConfig):
    """Used for simple tokenization tests
    """

    def get_generator_class(self):
        return None

    def get_training_callable(self):
        return None


@pytest.fixture(scope="module")
def input_data_path():
    return str(
        (Path(__file__).parent / "data" / "smol.txt").resolve()
    )

L1 = "Once upon a midnight dreary, while I pondered, weak and weary,\n"


def test_single_char(input_data_path, tmpdir):
    # NOTE: Here the line delim should not matter for this char tokenizer
    config = SimpleConfig(input_data_path=input_data_path, checkpoint_dir=tmpdir, field_delimiter=",")
    trainer = tok.CharTokenizerTrainer(config=config)

    # We need this for batch mode, so verify it can be copied
    deepcopy(trainer)

    line_iter = trainer.annotate_data()

    # Assert that we didn't do any annotation
    line_one = next(line_iter)
    assert line_one == L1

    # Let's train the tokenizer, and now reload it back in
    trainer.train()
    tokenizer = tok.CharTokenizer.load(tmpdir)
    assert tokenizer.total_vocab_size == 32

    # NOTE: this is because we default to using this token as a delim
    # in the main config, but this tokenizer doesn't do anything with it anyway
    assert tokenizer.field_delimiter == ","
    assert tokenizer.field_delimiter_token == "<d>"

    l1_ids = [6, 21, 11, 13, 1, 28, 23, 22, 21, 1, 9, 1, 20, 17, 12, 21, 17, 15, 16, 27, 1, 12, 25, 13, 9, 25, 31, 2, 1, 30, 16, 17, 19, 13, 1, 5, 1, 23, 22, 21, 12, 13, 25, 13, 12, 2, 1, 30, 13, 9, 18, 1, 9, 21, 12, 1, 30, 13, 9, 25, 31, 2, 0]
    assert tokenizer.encode_to_ids(L1) == l1_ids
    assert tokenizer.decode_from_ids(l1_ids) == L1

    # Check the factory
    assert isinstance(
        tok.tokenizer_from_model_dir(tmpdir),
        tok.CharTokenizer
    )


def test_single_char_small_vocab(input_data_path, tmpdir):
    config = SimpleConfig(input_data_path=input_data_path, checkpoint_dir=tmpdir)
    trainer = tok.CharTokenizerTrainer(config=config, vocab_size=10)
    trainer.annotate_data()
    
    trainer.train()
    tokenizer = tok.CharTokenizer.load(tmpdir)
    assert tokenizer.total_vocab_size == 10

    # Too small of a vocab...

    with pytest.raises(tok.TokenizerError):
        tokenizer.encode_to_ids("Once upon")

    with pytest.raises(tok.TokenizerError):
        tokenizer.decode_from_ids([11])


def test_sp(input_data_path, tmpdir):
    config = SimpleConfig(input_data_path=input_data_path, checkpoint_dir=tmpdir)
    trainer = tok.SentencePieceTokenizerTrainer(config=config)
    deepcopy(trainer)
    line_iter = trainer.annotate_data()

    line_one = next(line_iter)
    assert line_one == "Once upon a midnight dreary, while I pondered, weak and weary,<n>\n"

    trainer.train()
    tokenizer = tok.SentencePieceTokenizer.load(tmpdir)

    ids = [41, 54, 8, 5, 11, 36, 10, 14, 16, 13, 17, 16, 22, 20, 15, 5, 13, 25, 32, 7, 6, 51, 42, 9, 8, 5, 23, 5, 36, 13, 48, 13, 6, 49, 62, 10, 28, 49, 25, 7, 6, 3]
    assert tokenizer.encode_to_ids("Once upon a midnight dreary, while I pondered, weak and weary,<n>\n") == ids
    assert tokenizer.decode_from_ids(ids) == "Once upon a midnight dreary, while I pondered, weak and weary,<n>"


def test_sp_field_delim(input_data_path, tmpdir):
    config = SimpleConfig(input_data_path=input_data_path, checkpoint_dir=tmpdir, field_delimiter=",")
    trainer = tok.SentencePieceTokenizerTrainer(config=config)
    line_iter = trainer.annotate_data()

    line_one = next(line_iter)
    assert line_one == "Once upon a midnight dreary<d> while I pondered<d> weak and weary<d><n>\n"

    trainer.train()
    tokenizer = tok.SentencePieceTokenizer.load(tmpdir)

    ids = [40, 53, 7, 5, 10, 35, 9, 13, 15, 12, 16, 15, 21, 19, 14, 5, 12, 24, 30, 6, 4, 51, 41, 8, 7, 5, 23, 5, 35, 12, 47, 12, 4, 48, 61, 9, 27, 48, 24, 6, 4, 3]
    assert tokenizer.encode_to_ids("Once upon a midnight dreary<d> while I pondered<d> weak and weary<d><n>\n") == ids
    assert tokenizer.decode_from_ids(ids) == "Once upon a midnight dreary, while I pondered, weak and weary,<n>"

    # Check the factory
    assert isinstance(
        tok.tokenizer_from_model_dir(tmpdir),
        tok.SentencePieceTokenizer
    )
