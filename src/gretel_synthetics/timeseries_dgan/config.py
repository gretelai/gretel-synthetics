from dataclasses import asdict, dataclass
from enum import Enum


class OutputType(Enum):
    """Supported variables types.

    Determines internal representation of variables and output layers in
    generation network.
    """

    DISCRETE = 0
    CONTINUOUS = 1


class Normalization(Enum):
    """Normalization types for continuous variables.

    Determines if a sigmoid (ZERO_ONE) or tanh (MINUSONE_ONE) activation is used
    for the output layers in the generation network.
    """

    ZERO_ONE = 0
    MINUSONE_ONE = 1


@dataclass
class DGANConfig:
    """Config object with parameters for training a DGAN model.

    Args:
        max_sequence_len: length of time series sequences, variable length
            sequences are not supported, so all training and generated data will
            have the same length sequences
        sample_len: time series steps to generate from each LSTM cell in DGAN,
            must be a divisor of max_sequence_len
        attribute_noise_dim: length of the GAN noise vectors for attribute
            generation
        feature_noise_dim: length of GAN noise vectors for feature generation
        attribute_num_layers: # of layers in the GAN discriminator network
        attribute_num_units: # of units per layer in the GAN discriminator
            network
        feature_num_layers: # of LSTM layers in the GAN generator network
        feature_num_units: # of units per layer in the GAN generator network
        use_attribute_discriminator: use separaste discriminator only on
            attributes, helps DGAN match attribute distributions, Default: True
        normalization: default normalization for continuous variables, used when
            metadata output is not specified during DGAN initialization
        apply_feature_scaling: scale each continuous variable to [0,1] or [-1,1]
            (based on normalization param) before training and rescale to
            original range during generation, if False then training data must
            be within range and DGAN will only generate values in [0,1] or
            [-1,1], Default: True
        apply_example_scaling: compute midpoint and halfrange (equivalent to
            min/max) for each time series variable and include these as
            additional attributes that are generated, this provides better
            support for time series with highly variable ranges, e.g., in
            network data, a dial-up connection has bandwidth usage in [1kb,
            10kb], while a fiber connection is in [100mb, 1gb], Default: True
        binary_encoder_cutoff: use binary encoder (instead of one hot encoder) for
            any column with more than this many unique values. This helps reduce memory
            consumption for datasets with a lot of unique values.
        forget_bias: initialize forget gate bias paramters to 1 in LSTM layers,
            when True initialization matches tf1 LSTMCell behavior, otherwise
            default pytorch initialization is used, Default: False
        gradient_penalty_coef: coefficient for gradient penalty in Wasserstein
            loss, Default: 10.0
        attribute_gradient_penalty_coef: coefficient for gradient penalty in
            Wasserstein loss for the attribute discriminator, Default: 10.0
        attribute_loss_coef: coefficient for attribute discriminator loss in
            comparison the standard discriminator on attributes and features,
            higher values should encourage DGAN to learn attribute
            distributions, Default: 1.0
        generator_learning_rate: learning rate for Adam optimizer
        generator_beta1: Adam param for exponential decay of 1st moment
        discriminator_learning_rate: learning rate for Adam optimizer
        discriminator_beta1: Adam param for exponential decay of 1st moment
        attribute_discriminator_learning_rate: learning rate for Adam optimizer
        attribute_discriminator_beta1: Adam param for exponential decay of 1st
            moment
        batch_size: # of examples used in batches, for both training and
            generation
        epochs: # of epochs to train model discriminator_rounds: training steps
        for the discriminator(s) in each
            batch
        generator_rounds: training steps for the generator in each batch
        cuda: use GPU if available
        mixed_precision_training: enabling automatic mixed precision while training
            in order to reduce memory costs, bandwith, and time by identifying the
            steps that require full precision and using 32-bit floating point for
            only those steps while using 16-bit floating point everywhere else.
    """

    # Model structure
    max_sequence_len: int
    sample_len: int

    attribute_noise_dim: int = 10
    feature_noise_dim: int = 10
    attribute_num_layers: int = 3
    attribute_num_units: int = 100
    feature_num_layers: int = 1
    feature_num_units: int = 100
    use_attribute_discriminator: bool = True

    # Data transformation
    normalization: Normalization = Normalization.ZERO_ONE
    apply_feature_scaling: bool = True
    apply_example_scaling: bool = True
    binary_encoder_cutoff: int = 150

    # Model initialization
    forget_bias: bool = False

    # Loss function
    gradient_penalty_coef: float = 10.0
    attribute_gradient_penalty_coef: float = 10.0
    attribute_loss_coef: float = 1.0

    # Training
    generator_learning_rate: float = 0.001
    generator_beta1: float = 0.5
    discriminator_learning_rate: float = 0.001
    discriminator_beta1: float = 0.5
    attribute_discriminator_learning_rate: float = 0.001
    attribute_discriminator_beta1: float = 0.5
    batch_size: int = 1024
    epochs: int = 400
    discriminator_rounds: int = 1
    generator_rounds: int = 1

    cuda: bool = True
    mixed_precision_training: bool = False

    def to_dict(self):
        """Return dictionary representation of DGANConfig.

        Returns:
            Dictionary of member variables, usable to initialize a new config
            object, e.g., `DGANConfig(**config.to_dict())`
        """
        return asdict(self)


class DfStyle(str, Enum):
    """Supported styles for parsing pandas DataFrames.

    See `train_dataframe` method in dgan.py for details.
    """

    WIDE = "wide"
    LONG = "long"
