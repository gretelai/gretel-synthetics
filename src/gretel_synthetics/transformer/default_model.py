import logging


from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def build_default_model(store):
    """
    Load pre-trained model

    Args:
        store: LocalConfig

    Returns:
        transformers GPT2 model
    """

    # Set seed before initializing model
    set_seed(store.seed)

    config_kwargs = {
        "cache_dir": store.cache_dir,
        "revision": store.model_revision,
        "use_auth_token": store.use_auth_token
    }

    if store.config_name:
        config = AutoConfig.from_pretrained(store.config_name, **config_kwargs)
    elif store.model_name_or_path:
        config = AutoConfig.from_pretrained(store.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[store.model_type]()
        logging.warning("You are instantiating a new config instance from scratch.")

    if store.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            store.model_name_or_path,
            from_tf=bool(".ckpt" in store.model_name_or_path),
            config=config,
            cache_dir=store.cache_dir,
            revision=store.model_revision,
            use_auth_token=True if store.use_auth_token else None,
        )
    else:
        logging.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    return model
