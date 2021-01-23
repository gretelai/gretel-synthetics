"""
Initialize huggingface transformer model (GPT-2)
"""
from transformers import AutoTokenizer

from gretel_synthetics.transformer.default_model import build_default_model


def load_model(store, tokenizer: AutoTokenizer):
    """
    Build and return models in DP and non-DP (differentially private), if supported

    Args:
        store: Gretel-synthetics configuration class
        tokenizer: Huggingface generic tokenizer class

    Returns:
        transformers GPT2 Model
    """
    model = build_default_model(store)
    model.resize_token_embeddings(len(tokenizer))
    return model
