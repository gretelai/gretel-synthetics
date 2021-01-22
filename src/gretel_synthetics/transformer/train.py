"""
Train a GPT-2 based model to generate synthetic data
Based on https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py
"""
import logging
import math
import os
from typing import TYPE_CHECKING

from datasets import load_dataset, SplitGenerator, Split
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from gretel_synthetics.transformer.model import load_model

if TYPE_CHECKING:
    from gretel_synthetics.config_transformer import TransformerConfig
    from gretel_synthetics.train import TrainingParams
else:
    TransformerConfig = None
    TrainingParams = None


logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


def train(params: TrainingParams):
    """
    Fit synthetic data model on training data.

    Args:
        params: The parameters controlling model training.

    Returns:
        results: Dict of training results
    """
    store = params.config
    training_args = TrainingArguments(store.checkpoint_dir)
    training_args.do_train = store.do_train
    training_args.do_eval = store.do_eval
    training_args.num_train_epochs = store.epochs
    training_args.save_steps = store.save_steps
    training_args.seed = store.seed

    # Load a huggingface dataset from the hub
    if store.dataset_name is not None:
        datasets = load_dataset(store.dataset_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                store.dataset_name,
                split=f"train[:{store.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                store.dataset_name,
                split=f"train[{store.validation_split_percentage}%:]",
            )

    # Load a line delimited text dataset from disk
    elif store.input_data_path is not None:
        datasets = load_dataset("text", data_files=store.input_data_path)
        datasets["train"] = load_dataset("text",
                                         data_files=store.input_data_path,
                                         split=f"train[{store.validation_split_percentage}%:]")
        datasets["validation"] = load_dataset("text",
                                              data_files=store.input_data_path,
                                              split=f"train[:{store.validation_split_percentage}%]")
        print(datasets)
    else:
        logging.error("No dataset specified, exiting")
        raise RuntimeError

    # TODO: Should live in Tokenizers class
    tokenizer_kwargs = {
        "cache_dir": store.cache_dir,
        "use_fast": store.use_fast_tokenizer,
        "revision": store.model_revision,
        "use_auth_token": True if store.use_auth_token else None,
    }
    if store.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(store.tokenizer_name, **tokenizer_kwargs)
    elif store.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(store.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Instantiate and load GPT model
    model = load_model(store, tokenizer)

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Tokenize the texts
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=store.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not store.overwrite_cache,
    )

    if store.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logging.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if store.block_size > tokenizer.model_max_length:
            logging.warn(
                f"The block_size passed ({store.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(store.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset
    # and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=store.preprocessing_num_workers,
        load_from_cache_file=not store.overwrite_cache,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"] if training_args.do_train else None,
        eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Start training
    if training_args.do_train:
        model_path = (
            store.model_name_or_path
            if (store.model_name_or_path is not None and os.path.isdir(store.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logging.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logging.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logging.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_clm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logging.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logging.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

