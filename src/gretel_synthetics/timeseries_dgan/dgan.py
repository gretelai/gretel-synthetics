"""
PyTorch implementation of DoppelGANger, from https://arxiv.org/abs/1909.13403

Based on tensorflow 1 code in https://github.com/fjxmlzn/DoppelGANger

DoppelGANger is a generative adversarial network (GAN) model for time series. It
supports multi-variate time series (referred to as features) and fixed variables
for each time series (attributes). The combination of attribute values and
sequence of feature values is 1 example. Once trained, the model can generate
novel examples that exhibit the same temporal correlations as seen in the
training data. See https://arxiv.org/abs/1909.13403 for additional details on
the model.

As a reference for terminology, consider open-high-low-close (OHLC) data from
stock markets. Each stock is an example, with fixed attributes such as exchange,
sector, country. The features or time series consists of open, high, low, and
closing prices for each time interval (daily). After being trained on historical
data, the model can generate more hypothetical stocks and price behavior on the
training time range.


Sample usage:

.. code-block::

   import numpy as np
   from gretel_synthetics.timeseries_dgan.dgan import DGAN
   from gretel_synthetics.timeseries_dgan.config import DGANConfig

   attributes = np.random.rand(10000, 3)
   features = np.random.rand(10000, 20, 2)

   config = DGANConfig(
       max_sequence_len=20,
       sample_len=5,
       batch_size=1000,
       epochs=10
   )

   model = DGAN(config)

   model.train_numpy(attributes, features)

   synthetic_attributes, synthetic_features = model.generate_numpy(1000)
"""


from __future__ import annotations

import abc
import logging
import math

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gretel_synthetics.timeseries_dgan.config import DfStyle, DGANConfig, OutputType
from gretel_synthetics.timeseries_dgan.torch_modules import Discriminator, Generator
from gretel_synthetics.timeseries_dgan.transformations import (
    create_additional_attribute_outputs,
    create_outputs_from_data,
    inverse_transform,
    Output,
    transform,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)

AttributeFeaturePair = Tuple[Optional[np.ndarray], np.ndarray]
NumpyArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]

NAN_ERROR_MESSAGE = """
DGAN does not support NaNs, please remove NaNs before training. If there are no NaNs in your input data and you see this error, please create a support ticket.
"""


class DGAN:
    """
    DoppelGANger model.

    Interface for training model and generating data based on configuration in
    an DGANConfig instance.

    DoppelGANger uses a specific internal representation for data which is
    hidden from the user in the public interface. Continuous variables should be
    in the original space and discrete variables represented as [0.0, 1.0, 2.0,
    ...] when using the train_numpy() and train_dataframe() functions. The
    generate_numpy() and generate_dataframe() functions will return data in this
    original space. In standard usage, the detailed transformation info in
    attribute_outputs and feature_outputs are not needed, those will be created
    automatically when a train* function is called with data.

    If more control is needed and you want to use the normalized values and
    one-hot encoding directly, use the _train() and _generate() functions.
    transformations.py contains internal helper functions for working with the
    Output metadata instances and converting data to and from the internal
    representation. To dive even deeper into the model structure, see the
    torch_modules.py which contains the torch implementations of the networks
    used in DGAN. As internal details, transformations.py and torch_modules.py
    are not part of the public interface and may change at any time without
    notice.

    """

    def __init__(
        self,
        config: DGANConfig,
        attribute_outputs: Optional[List[Output]] = None,
        feature_outputs: Optional[List[Output]] = None,
    ):
        """Create a DoppelGANger model.

        Args:
            config: DGANConfig containing model parameters
            attribute_outputs: custom metadata for attributes, not needed for
                standard usage
            feature_outputs: custom metadata for features, not needed for
                standard usage
        """
        self.config = config

        self.is_built = False

        if config.max_sequence_len % config.sample_len != 0:
            raise RuntimeError(
                f"max_sequence_len={config.max_sequence_len} must be divisible by sample_len={config.sample_len}"
            )

        if feature_outputs is not None and attribute_outputs is not None:
            self._build(attribute_outputs, feature_outputs)
        elif feature_outputs is not None or attribute_outputs is not None:
            raise RuntimeError(
                "feature_outputs and attribute_ouputs must either both be given or both be None"
            )

        self.data_frame_converter = None

    def train_numpy(
        self,
        features: np.ndarray,
        feature_types: Optional[List[OutputType]] = None,
        attributes: Optional[np.ndarray] = None,
        attribute_types: Optional[List[OutputType]] = None,
    ):
        """Train DGAN model on data in numpy arrays.

        Training data is passed in 2 numpy arrays, one for attributes (2d) and
        one for features (3d). This data should be in the original space and is
        not transformed. If the data is already transformed into the internal
        DGAN representation (continuous variable scaled to [0,1] or [-1,1] and
        discrete variables one-hot encoded), use the internal _train() function
        instead of train_numpy(), or specify apply_feature_scaling=False in the
        DGANConfig.

        In standard usage, attribute_types and feature_types should be provided
        on the first call to train() to correctly setup the model structure. If
        not specified, the default is to assume continuous variables. If outputs
        metadata was specified when the instance was initialized or train() was
        previously called, then attribute_types and feature_types are not
        needed.

        Args:
            features: 3-d numpy array of time series features for the training,
                size is (# of training examples) X max_sequence_len X (# of
                features)
            feature_types (Optional): Specification of Discrete or Continuous type
                for each variable of the features. Discrete attributes should be
                0-indexed (not one-hot encoded). If None, assume all features
                are continuous. Ignored if the model was already built, either
                by passing *output params at initialization or because train_*
                was called previously.
            attributes (Optional): 2-d numpy array of attributes for the training
                examples, size is (# of training examples) X (# of attributes)
            attribute_types (Optional): Specification of Discrete or Continuous
                type for each variable of the attributes. Discrete attributes
                should be 0-indexed (not one-hot encoded). If None, assume all
                attributes are continuous. Ignored if the model was already
                built, either by passing *output params at initialization or
                because train_* was called previously.
        """
        if attributes is not None:
            if attributes.shape[0] != features.shape[0]:
                raise RuntimeError(
                    "First dimension of attributes and features must be the same length, i.e., the number of training examples."
                )

        if attributes is not None and attribute_types is None:
            # Automatically determine attribute types
            attribute_types = []
            for i in range(attributes.shape[1]):
                try:
                    # Here we treat integer columns as continuous, and thus the
                    # generated values will be (unrounded) floats. This may not
                    # be the right choice, and may be surprising to give integer
                    # inputs and get back floats. An explicit list of
                    # feature_types can be given (or constructed by passing
                    # discrete_columns to train_dataframe) to control this
                    # behavior. And we can look into a better fix in the future,
                    # maybe using # of distinct values, and having an explicit
                    # integer type so we appropriately round the final output.
                    attributes[:, i].astype("float")
                    attribute_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    attribute_types.append(OutputType.DISCRETE)

        if feature_types is None:
            # Automatically determine feature types
            feature_types = []
            for i in range(features.shape[2]):
                try:
                    # Here we treat integer columns as continuous, see above
                    # comment.
                    features[:, :, i].astype("float")
                    feature_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    feature_types.append(OutputType.DISCRETE)

        if not self.is_built:
            attribute_outputs, feature_outputs = create_outputs_from_data(
                attributes,
                features,
                attribute_types,
                feature_types,
                normalization=self.config.normalization,
                apply_feature_scaling=self.config.apply_feature_scaling,
                apply_example_scaling=self.config.apply_example_scaling,
            )

            self._build(
                attribute_outputs,
                feature_outputs,
            )

        continuous_features_ind = [
            ind
            for ind, val in enumerate(self.feature_outputs)
            if "ContinuousOutput" in str(val.__class__)
        ]

        valid_examples = validation_check(
            features[:, :, continuous_features_ind].astype("float")
        )
        # Only using valid examples for the entire dataset.
        features = features[valid_examples]
        # Apply linear interpolations for continuous features:
        features[:, :, continuous_features_ind] = nan_linear_interpolation(
            features[:, :, continuous_features_ind].astype("float")
        )

        if attributes is not None:
            attributes = attributes[valid_examples]

        if self.additional_attribute_outputs:
            (
                internal_features,
                internal_additional_attributes,
            ) = transform(features, self.feature_outputs, variable_dim_index=2)

            if np.any(np.isnan(internal_additional_attributes)):
                raise ValueError(
                    f"NaN found in internal additional attributes. {NAN_ERROR_MESSAGE}"
                )

        else:
            internal_features = transform(
                features, self.feature_outputs, variable_dim_index=2
            )
            internal_additional_attributes = torch.Tensor(
                np.full((internal_features.shape[0], 1), np.nan)
            )

        internal_attributes = transform(
            attributes,
            self.attribute_outputs,
            variable_dim_index=1,
            num_examples=internal_features.shape[0],
        )

        if self.attribute_outputs and np.any(np.isnan(internal_attributes)):
            raise ValueError(f"NaN found in internal attributes. {NAN_ERROR_MESSAGE}")

        dataset = TensorDataset(
            torch.Tensor(internal_attributes),
            torch.Tensor(internal_additional_attributes),
            torch.Tensor(internal_features),
        )

        self._train(dataset)

    def train_dataframe(
        self,
        df: pd.DataFrame,
        attribute_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        example_id_column: Optional[str] = None,
        time_column: Optional[str] = None,
        discrete_columns: Optional[List[str]] = None,
        df_style: DfStyle = DfStyle.WIDE,
    ):
        """Train DGAN model on data in pandas DataFrame.

        Training data can be in either "wide" or "long" format. "Wide" format
        uses one row for each example with 0 or more attribute columns and 1
        column per time point in the time series. "Wide" format is restricted to
        1 feature variable. "Long" format uses one row per time point, supports
        multiple feature variables, and uses additional example id to split into
        examples and time column to sort.

        Args:
            df: DataFrame of training data
            attribute_columns: list of column names containing attributes, if None,
                no attribute columns are used, Default: None
            feature_columns: list of column names containing features, if None
                all non-attribute columns are used, Default: None
            example_id_column: column name used to split "long" format data
                frame into multiple examples, if None, data is treated as a
                single example
            time_column: column name used to sort "long" format data frame,
                if None, data frame order of rows/time points is used
            discrete_columns: column names (either attributes or features) to
                use discrete, onehot encoding, discrete values must be integer
                in [0,1,2,3...]
            df_style: str enum of "wide" or "long" indicating format of the
                DataFrame
        """

        if self.data_frame_converter is None:

            if df_style == DfStyle.WIDE:
                self.data_frame_converter = _WideDataFrameConverter.create(
                    df,
                    attribute_columns=attribute_columns,
                    feature_columns=feature_columns,
                    discrete_columns=discrete_columns,
                )
            elif df_style == DfStyle.LONG:
                if example_id_column == None and attribute_columns:
                    raise Exception(
                        "Please provide an example id column, auto-splitting not available with only attribute columns."
                    )
                if example_id_column == None and attribute_columns == None:
                    logging.warning(
                        f"Example ID column not provided, DGAN will autosplit dataset into sequences of size {self.config.max_sequence_len}!"
                    )
                    df = df[
                        : math.floor(len(df) / self.config.max_sequence_len)
                        * self.config.max_sequence_len
                    ].copy()
                    if time_column != None:
                        df[time_column] = pd.to_datetime(df[time_column])
                        df = df.sort_values(time_column)
                    example_id_column = "example_id"
                    df[example_id_column] = np.repeat(
                        range(len(df) // self.config.max_sequence_len),
                        self.config.max_sequence_len,
                    )

                self.data_frame_converter = _LongDataFrameConverter.create(
                    df,
                    attribute_columns=attribute_columns,
                    feature_columns=feature_columns,
                    example_id_column=example_id_column,
                    time_column=time_column,
                    discrete_columns=discrete_columns,
                )
            else:
                raise ValueError(
                    f"df_style param must be an enum value DfStyle ('wide' or 'long'), received '{df_style}'"
                )

        attributes, features = self.data_frame_converter.convert(df)

        self.train_numpy(
            attributes=attributes,
            features=features,
            attribute_types=self.data_frame_converter.attribute_types,
            feature_types=self.data_frame_converter.feature_types,
        )

    def generate_numpy(
        self,
        n: Optional[int] = None,
        attribute_noise: Optional[torch.Tensor] = None,
        feature_noise: Optional[torch.Tensor] = None,
    ) -> AttributeFeaturePair:
        """Generate synthetic data from DGAN model.

        Once trained, a DGAN model can generate arbitrary amounts of
        synthetic data by sampling from the noise distributions. Specify either
        the number of records to generate, or the specific noise vectors to use.

        Args:
            n: number of examples to generate
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            Tuple of attributes and features as numpy arrays.
        """

        if n is not None:
            # Generate across multiple batches of batch_size. Use same size for
            # all batches and truncate the last partial batch at the very end
            # before returning.
            num_batches = n // self.config.batch_size
            if n % self.config.batch_size != 0:
                num_batches += 1

            internal_data_list = []
            for _ in range(num_batches):
                internal_data_list.append(
                    self._generate(
                        self.attribute_noise_func(self.config.batch_size),
                        self.feature_noise_func(self.config.batch_size),
                    )
                )
            # Convert from list of tuples to tuple of lists with zip(*) and
            # concatenate into single numpy arrays for attributes, additional
            # attributes (if present), and features.
            internal_data = tuple(
                np.concatenate(d, axis=0) if not (np.array(d) == None).any() else None
                for d in zip(*internal_data_list)
            )

        else:
            if attribute_noise is None or feature_noise is None:
                raise RuntimeError(
                    "generate() must receive either n or both attribute_noise and feature_noise"
                )
            attribute_noise = attribute_noise.to(self.device, non_blocking=True)
            feature_noise = feature_noise.to(self.device, non_blocking=True)

            internal_data = self._generate(attribute_noise, feature_noise)

        (
            internal_attributes,
            internal_additional_attributes,
            internal_features,
        ) = internal_data

        attributes = inverse_transform(
            internal_attributes, self.attribute_outputs, variable_dim_index=1
        )

        features = inverse_transform(
            internal_features,
            self.feature_outputs,
            variable_dim_index=2,
            additional_attributes=internal_additional_attributes,
        )

        if n is not None:
            if attributes is None:
                features = features[:n]
                return None, features
            else:
                return attributes[:n], features[:n]

        return attributes, features

    def generate_dataframe(
        self,
        n: Optional[int] = None,
        attribute_noise: Optional[torch.Tensor] = None,
        feature_noise: Optional[torch.Tensor] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data from DGAN model.

        Once trained, a DGAN model can generate arbitrary amounts of
        synthetic data by sampling from the noise distributions. Specify either
        the number of records to generate, or the specific noise vectors to use.

        Args:
            n: number of examples to generate
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            pandas DataFrame in same format used in 'train_dataframe' call
        """

        attributes, features = self.generate_numpy(n, attribute_noise, feature_noise)

        return self.data_frame_converter.invert(attributes, features)

    def _build(
        self,
        attribute_outputs: Optional[List[Output]],
        feature_outputs: List[Output],
    ):
        """Setup internal structure for DGAN model.

        Args:
            attribute_outputs: custom metadata for attributes
            feature_outputs: custom metadata for features
        """

        self.EPS = 1e-8
        self.attribute_outputs = attribute_outputs
        self.additional_attribute_outputs = create_additional_attribute_outputs(
            feature_outputs
        )
        self.feature_outputs = feature_outputs

        if self.config.cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.generator = Generator(
            attribute_outputs,
            self.additional_attribute_outputs,
            feature_outputs,
            self.config.max_sequence_len,
            self.config.sample_len,
            self.config.attribute_noise_dim,
            self.config.feature_noise_dim,
            self.config.attribute_num_units,
            self.config.attribute_num_layers,
            self.config.feature_num_units,
            self.config.feature_num_layers,
        )

        self.generator.to(self.device, non_blocking=True)

        if self.attribute_outputs is None:
            self.attribute_outputs = []
        attribute_dim = sum(output.dim for output in self.attribute_outputs)

        if not self.additional_attribute_outputs:
            self.additional_attribute_outputs = []
        additional_attribute_dim = sum(
            output.dim for output in self.additional_attribute_outputs
        )
        feature_dim = sum(output.dim for output in feature_outputs)
        self.feature_discriminator = Discriminator(
            attribute_dim
            + additional_attribute_dim
            + self.config.max_sequence_len * feature_dim,
            num_layers=5,
            num_units=200,
        )
        self.feature_discriminator.to(self.device, non_blocking=True)

        self.attribute_discriminator = None
        if not self.additional_attribute_outputs and not self.attribute_outputs:
            self.config.use_attribute_discriminator = False

        if self.config.use_attribute_discriminator:
            self.attribute_discriminator = Discriminator(
                attribute_dim + additional_attribute_dim,
                num_layers=5,
                num_units=200,
            )
            self.attribute_discriminator.to(self.device, non_blocking=True)

        self.attribute_noise_func = lambda batch_size: torch.randn(
            batch_size, self.config.attribute_noise_dim, device=self.device
        )

        self.feature_noise_func = lambda batch_size: torch.randn(
            batch_size,
            self.config.max_sequence_len // self.config.sample_len,
            self.config.feature_noise_dim,
            device=self.device,
        )

        if self.config.forget_bias:

            def init_weights(m):
                if "LSTM" in str(m.__class__):
                    for name, param in m.named_parameters(recurse=False):
                        if "bias_hh" in name:
                            # The LSTM bias param is a concatenation of 4 bias
                            # terms: (b_ii|b_if|b_ig|b_io). We only want to
                            # change the forget gate bias, i.e., b_if. But we
                            # can't change a slice of the tensor, so need to
                            # recreate the initialization for the other parts
                            # and concatenate with the new forget gate bias
                            # initialization.
                            with torch.no_grad():
                                hidden_size = m.hidden_size
                                a = -np.sqrt(1.0 / hidden_size)
                                b = np.sqrt(1.0 / hidden_size)
                                bias_ii = torch.Tensor(hidden_size)
                                bias_ig_io = torch.Tensor(hidden_size * 2)
                                bias_if = torch.Tensor(hidden_size)
                                torch.nn.init.uniform_(bias_ii, a, b)
                                torch.nn.init.uniform_(bias_ig_io, a, b)
                                torch.nn.init.ones_(bias_if)
                                new_param = torch.cat(
                                    [bias_ii, bias_if, bias_ig_io], dim=0
                                )
                                param.copy_(new_param)

            self.generator.apply(init_weights)

        self.is_built = True

    def _train(
        self,
        dataset: Dataset,
    ):
        """Internal method for training DGAN model.

        Expects data to already be transformed into the internal representation
        and wrapped in a torch Dataset. The torch Dataset consists of 3-element
        tuples (attributes, additional_attributes, features). If attributes and/or
        additional_attribtues were not passed by the user, these indexes of the
        tuple will consists of nan-filled tensors which will later be filtered
        out and ignored in the DGAN training process.

        Args:
            dataset: torch Dataset containing tuple of (attributes, additional_attributes, features)
        """
        if len(dataset) <= 1:
            raise ValueError(
                f"DGAN requires multiple examples to train, received {len(dataset)} example."
                + "Consider splitting a single long sequence into many subsequences to obtain "
                + "multiple examples for training."
            )

        # Our optimization setup does not work on batches of size 1. So if
        # drop_last=False would produce a last batch of size of 1, we use
        # drop_last=True instead.
        drop_last = len(dataset) % self.config.batch_size == 1

        loader = DataLoader(
            dataset,
            self.config.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=2,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            multiprocessing_context="fork",
        )

        opt_discriminator = torch.optim.Adam(
            self.feature_discriminator.parameters(),
            lr=self.config.discriminator_learning_rate,
            betas=(self.config.discriminator_beta1, 0.999),
        )

        opt_attribute_discriminator = None
        if self.attribute_discriminator is not None:
            opt_attribute_discriminator = torch.optim.Adam(
                self.attribute_discriminator.parameters(),
                lr=self.config.attribute_discriminator_learning_rate,
                betas=(self.config.attribute_discriminator_beta1, 0.999),
            )

        opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.generator_learning_rate,
            betas=(self.config.generator_beta1, 0.999),
        )

        global_step = 0

        # Set torch modules to training mode
        self._set_mode(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision_training)

        for epoch in range(self.config.epochs):
            logger.info(f"epoch: {epoch}")

            for real_batch in loader:
                global_step += 1

                with torch.cuda.amp.autocast(
                    enabled=self.config.mixed_precision_training
                ):
                    attribute_noise = self.attribute_noise_func(real_batch[0].shape[0])
                    feature_noise = self.feature_noise_func(real_batch[0].shape[0])

                    # Both real and generated batch are always three element tuple of
                    # tensors. The tuple is structured as follows: (attribute_output,
                    # additional_attribute_output, feature_output). If self.attribute_output
                    # and/or self.additional_attribute_output is empty, the respective
                    # tuple index will be filled with a placeholder nan-filled tensor.
                    # These nan-filled tensors get filtered out in the _discriminate,
                    # _get_gradient_penalty, and _discriminate_attributes functions.

                    generated_batch = self.generator(attribute_noise, feature_noise)
                    real_batch = [
                        x.to(self.device, non_blocking=True) for x in real_batch
                    ]

                for _ in range(self.config.discriminator_rounds):
                    opt_discriminator.zero_grad(
                        set_to_none=self.config.mixed_precision_training
                    )
                    with torch.cuda.amp.autocast(enabled=True):
                        generated_output = self._discriminate(generated_batch)
                        real_output = self._discriminate(real_batch)

                        loss_generated = torch.mean(generated_output)
                        loss_real = -torch.mean(real_output)
                        loss_gradient_penalty = self._get_gradient_penalty(
                            generated_batch, real_batch, self._discriminate
                        )

                        loss = (
                            loss_generated
                            + loss_real
                            + self.config.gradient_penalty_coef * loss_gradient_penalty
                        )

                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(opt_discriminator)
                    scaler.update()

                    if opt_attribute_discriminator is not None:
                        opt_attribute_discriminator.zero_grad(set_to_none=False)
                        # Exclude features (last element of batches) for
                        # attribute discriminator
                        with torch.cuda.amp.autocast(
                            enabled=self.config.mixed_precision_training
                        ):
                            generated_output = self._discriminate_attributes(
                                generated_batch[:-1]
                            )
                            real_output = self._discriminate_attributes(real_batch[:-1])

                            loss_generated = torch.mean(generated_output)
                            loss_real = -torch.mean(real_output)
                            loss_gradient_penalty = self._get_gradient_penalty(
                                generated_batch[:-1],
                                real_batch[:-1],
                                self._discriminate_attributes,
                            )

                            attribute_loss = (
                                loss_generated
                                + loss_real
                                + self.config.attribute_gradient_penalty_coef
                                * loss_gradient_penalty
                            )

                        scaler.scale(attribute_loss).backward(retain_graph=True)
                        scaler.step(opt_attribute_discriminator)
                        scaler.update()

                for _ in range(self.config.generator_rounds):
                    opt_generator.zero_grad(set_to_none=False)
                    with torch.cuda.amp.autocast(
                        enabled=self.config.mixed_precision_training
                    ):
                        generated_output = self._discriminate(generated_batch)

                        if self.attribute_discriminator:
                            # Exclude features (last element of batch) before
                            # calling attribute discriminator
                            attribute_generated_output = self._discriminate_attributes(
                                generated_batch[:-1]
                            )

                            loss = -torch.mean(
                                generated_output
                            ) + self.config.attribute_loss_coef * -torch.mean(
                                attribute_generated_output
                            )
                        else:
                            loss = -torch.mean(generated_output)

                    scaler.scale(loss).backward()
                    scaler.step(opt_generator)
                    scaler.update()

    def _generate(
        self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor
    ) -> NumpyArrayTriple:
        """Internal method for generating from a DGAN model.

        Returns data in the internal representation, including additional
        attributes for the midpoint and half-range for features when
        apply_example_scaling is True for some features.

        Args:
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            Tuple of generated data in internal representation. If additional
            attributes are used in the model, the tuple is 3 elements:
            attributes, additional_attributes, features. If there are no
            additional attributes in the model, the tuple is 2 elements:
            attributes, features.
        """
        # Set torch modules to eval mode
        self._set_mode(False)
        batch = self.generator(attribute_noise, feature_noise)
        return tuple(t.cpu().detach().numpy() for t in batch)

    def _discriminate(
        self,
        batch,
    ) -> torch.Tensor:
        """Internal helper function to apply the GAN discriminator.

        Args:
            batch: internal data representation

        Returns:
            Output of the GAN discriminator.
        """

        batch = [index for index in batch if not torch.isnan(index).any()]
        inputs = list(batch)
        # Flatten the features

        inputs[-1] = torch.reshape(inputs[-1], (inputs[-1].shape[0], -1))

        input = torch.cat(inputs, dim=1)

        output = self.feature_discriminator(input)
        return output

    def _discriminate_attributes(self, batch) -> torch.Tensor:
        """Internal helper function to apply the GAN attribute discriminator.

        Args:
            batch: tuple of internal data of size 2 elements
            containing attributes and additional_attributes.

        Returns:
            Output for GAN attribute discriminator.
        """
        batch = [index for index in batch if not torch.isnan(index).any()]
        if not self.attribute_discriminator:
            raise RuntimeError(
                "discriminate_attributes called with no attribute_discriminator"
            )

        input = torch.cat(batch, dim=1)

        output = self.attribute_discriminator(input)
        return output

    def _get_gradient_penalty(
        self, generated_batch, real_batch, discriminator_func
    ) -> torch.Tensor:
        """Internal helper function to compute the gradient penalty component of
        DoppelGANger loss.

        Args:
            generated_batch: internal data from the generator
            real_batch: internal data for the training batch
            discriminator_func: function to apply discriminator to interpolated
                data

        Returns:
            Gradient penalty tensor.
        """
        generated_batch = [
            generated_index
            for generated_index in generated_batch
            if not torch.isnan(generated_index).any()
        ]
        real_batch = [
            real_index for real_index in real_batch if not torch.isnan(real_index).any()
        ]

        alpha = torch.rand(generated_batch[0].shape[0], device=self.device)
        interpolated_batch = [
            self._interpolate(g, r, alpha).requires_grad_(True)
            for g, r in zip(generated_batch, real_batch)
        ]

        interpolated_output = discriminator_func(interpolated_batch)

        gradients = torch.autograd.grad(
            interpolated_output,
            interpolated_batch,
            grad_outputs=torch.ones(interpolated_output.shape, device=self.device),
            retain_graph=True,
            create_graph=True,
        )

        squared_sums = [
            torch.sum(torch.square(g.view(g.size(0), -1))) for g in gradients
        ]

        norm = torch.sqrt(sum(squared_sums) + self.EPS)

        return ((norm - 1.0) ** 2).mean()

    def _interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Internal helper function to interpolate between 2 tensors.

        Args:
            x1: tensor
            x2: tensor
            alpha: scale or 1d tensor with values in [0,1]

        Returns:
            x1 + alpha * (x2 - x1)
        """
        diff = x2 - x1
        expanded_dims = [1 for _ in diff.shape]
        expanded_dims[0] = -1
        reshaped_alpha = alpha.reshape(expanded_dims).expand(diff.shape)

        return x1 + reshaped_alpha * diff

    def _set_mode(self, mode: bool = True):
        """Set torch module training mode.

        Args:
            train_mode: whether to set training mode (True) or evaluation mode
                (False). Default: True
        """
        self.generator.train(mode)
        self.feature_discriminator.train(mode)
        if self.attribute_discriminator:
            self.attribute_discriminator.train(mode)

    def save(self, file_name: str, **kwargs):
        """Save DGAN model to a file.

        Args:
            file_name: location to save serialized model
            kwargs: additional parameters passed to torch.save
        """
        state = {
            "config": self.config.to_dict(),
            "attribute_outputs": self.attribute_outputs,
            "feature_outputs": self.feature_outputs,
        }
        state["generate_state_dict"] = self.generator.state_dict()
        state[
            "feature_discriminator_state_dict"
        ] = self.feature_discriminator.state_dict()
        if self.attribute_discriminator is not None:
            state[
                "attribute_discriminator_state_dict"
            ] = self.attribute_discriminator.state_dict()

        if self.data_frame_converter is not None:
            state["data_frame_converter"] = self.data_frame_converter.state_dict()

        torch.save(state, file_name, **kwargs)

    @classmethod
    def load(cls, file_name: str, **kwargs) -> DGAN:
        """Load DGAN model instance from a file.

        Args:
            file_name: location to load from
            kwargs: additional parameters passed to torch.load, for example, use
                map_location=torch.device("cpu") to load a model saved for GPU on
                a machine without cuda

        Returns:
            DGAN model instance
        """

        state = torch.load(file_name, **kwargs)

        config = DGANConfig(**state["config"])
        dgan = DGAN(config)

        dgan._build(state["attribute_outputs"], state["feature_outputs"])

        dgan.generator.load_state_dict(state["generate_state_dict"])
        dgan.feature_discriminator.load_state_dict(
            state["feature_discriminator_state_dict"]
        )
        if "attribute_discriminator_state_dict" in state:
            if dgan.attribute_discriminator is None:
                raise RuntimeError(
                    "Error deserializing model: found unexpected attribute discriminator state in file"
                )

            dgan.attribute_discriminator.load_state_dict(
                state["attribute_discriminator_state_dict"]
            )

        if "data_frame_converter" in state:
            dgan.data_frame_converter = _DataFrameConverter.load_from_state_dict(
                state["data_frame_converter"]
            )

        return dgan


class _DataFrameConverter(abc.ABC):
    """Abstract class for converting DGAN input to and from a DataFrame."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Class name used for serialization."""
        ...

    @property
    @abc.abstractmethod
    def attribute_types(self) -> List[OutputType]:
        """Output types used for attributes."""
        ...

    @property
    @abc.abstractmethod
    def feature_types(self) -> List[OutputType]:
        """Output types used for features."""
        ...

    @abc.abstractmethod
    def convert(self, df: pd.DataFrame) -> AttributeFeaturePair:
        """Convert DataFrame to DGAN input format.

        Args:
            df: DataFrame of training data

        Returns:
            Attribute (optional) and feature numpy arrays.
        """
        ...

    @abc.abstractmethod
    def invert(
        self, attributes: Optional[np.ndarray], features: np.ndarray
    ) -> pd.DataFrame:
        """Invert from DGAN input format back to DataFrame.

        Args:
            attributes: 2d numpy array of attributes
            features: 3d numpy array of features

        Returns:
            DataFrame representing attributes and features in original format.
        """
        ...

    def state_dict(self) -> Dict:
        """Dictionary describing this converter to use in saving and loading."""
        state = self._state_dict()
        state["name"] = self.name
        return state

    @abc.abstractmethod
    def _state_dict() -> Dict:
        """Subclass specific dictionary for saving and loading."""
        ...

    @classmethod
    def load_from_state_dict(cls, state: Dict) -> _DataFrameConverter:
        """Load a converter previously saved to a state dictionary."""
        # Assumes saved state was created with `state_dict()` method with name
        # and other params to initialize the class specified in
        # CONVERTER_CLASS_MAP. Care is required when modifying constructor
        # params or changing names if backwards compatibility is required.
        sub_class = CONVERTER_CLASS_MAP[state.pop("name")]

        return sub_class(**state)


class _WideDataFrameConverter(_DataFrameConverter):
    """Convert "wide" format DataFrames.

    Expects one row for each example with 0 or more attribute columns and 1
    column per time point in the time series.
    """

    def __init__(
        self,
        attribute_columns: List[str],
        feature_columns: List[str],
        discrete_columns: List[str],
        df_column_order: List[str],
        attribute_types: List[OutputType],
        feature_types: List[OutputType],
    ):
        super().__init__()
        self._attribute_columns = attribute_columns
        self._feature_columns = feature_columns
        self._discrete_columns = discrete_columns
        self._df_column_order = df_column_order
        self._attribute_types = attribute_types
        self._feature_types = feature_types

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        attribute_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        discrete_columns: Optional[List[str]] = None,
    ) -> _WideDataFrameConverter:
        """Create a converter instance.

        See `train_dataframe` for parameter details.
        """
        if attribute_columns is None:
            attribute_columns = []
        else:
            attribute_columns = attribute_columns

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in attribute_columns]
        else:
            feature_columns = feature_columns

        df_column_order = [
            c for c in df.columns if c in attribute_columns or c in feature_columns
        ]

        if discrete_columns is None:
            discrete_column_set = set()
        else:
            discrete_column_set = set(discrete_columns)

        # Check for string columns and ensure they are considered discrete.
        for column_name in df.columns:
            if df[column_name].dtype == "O":
                logging.info(
                    f"Marking column {column_name} as discrete because its type is string/object."
                )
                discrete_column_set.add(column_name)

        attribute_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in attribute_columns
        ]
        # With wide format, there's always 1 feature. It's only discrete if
        # every column used (every time point) is discrete.
        if all(c in discrete_column_set for c in feature_columns):
            feature_types = [OutputType.DISCRETE]
        else:
            feature_types = [OutputType.CONTINUOUS]

        return _WideDataFrameConverter(
            attribute_columns=attribute_columns,
            feature_columns=feature_columns,
            discrete_columns=sorted(discrete_column_set),
            df_column_order=df_column_order,
            attribute_types=attribute_types,
            feature_types=feature_types,
        )

    @property
    def name(self) -> str:
        return "WideDataFrameConverter"

    @property
    def attribute_types(self):
        return self._attribute_types

    @property
    def feature_types(self):
        return self._feature_types

    def convert(self, df: pd.DataFrame) -> AttributeFeaturePair:
        if self._attribute_columns:
            attributes = df[self._attribute_columns].to_numpy()
        else:
            attributes = None

        features = np.expand_dims(df[self._feature_columns].to_numpy(), axis=-1)

        return attributes, features

    def invert(
        self, attributes: Optional[np.ndarray], features: np.ndarray
    ) -> pd.DataFrame:
        if self._attribute_columns:
            if attributes is None:
                raise ValueError(
                    "Data converter with attribute columns expects attributes array, received None"
                )
            data = np.concatenate(
                (attributes, features.reshape(features.shape[0], features.shape[1])),
                axis=1,
            )
        else:
            data = features.reshape(features.shape[0], features.shape[1])

        df = pd.DataFrame(data, columns=self._attribute_columns + self._feature_columns)

        # Convert discrete columns to int where possible.
        for c in self._discrete_columns:
            try:
                df[c] = df[c].astype("int")
            except ValueError:
                pass

        # Ensure we match the original ordering
        return df[self._df_column_order]

    def _state_dict(self) -> Dict:
        return {
            "attribute_columns": self._attribute_columns,
            "feature_columns": self._feature_columns,
            "discrete_columns": self._discrete_columns,
            "df_column_order": self._df_column_order,
            "attribute_types": self._attribute_types,
            "feature_types": self._feature_types,
        }


class _LongDataFrameConverter(_DataFrameConverter):
    """Convert "long" format DataFrames.

    Expects one row per time point. Splits into examples based on specified
    example id column.
    """

    def __init__(
        self,
        attribute_columns: List[str],
        feature_columns: List[str],
        example_id_column: Optional[str],
        time_column: Optional[str],
        discrete_columns: List[str],
        df_column_order: List[str],
        attribute_types: List[OutputType],
        feature_types: List[OutputType],
        time_column_values: Optional[List[str]],
    ):
        super().__init__()
        self._attribute_columns = attribute_columns
        self._feature_columns = feature_columns
        self._example_id_column = example_id_column
        self._time_column = time_column
        self._discrete_columns = discrete_columns
        self._df_column_order = df_column_order
        self._attribute_types = attribute_types
        self._feature_types = feature_types
        self._time_column_values = time_column_values

    @classmethod
    def create(
        cls,
        df: pd.DataFrame,
        attribute_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        example_id_column: Optional[str] = None,
        time_column: Optional[str] = None,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Create a converter instance.

        See `train_dataframe` for parameter details.
        """
        if attribute_columns is None:
            attribute_columns = []
        else:
            attribute_columns = attribute_columns

        given_columns = set(attribute_columns)
        if example_id_column is not None:
            given_columns.add(example_id_column)
        if time_column is not None:
            given_columns.add(time_column)

        if feature_columns is None:
            # If not specified, use remaining columns in the data frame that
            # are not used elsewhere
            feature_columns = [c for c in df.columns if c not in given_columns]
        else:
            feature_columns = feature_columns

        # Add feature columns too, so given_columns contains all columns of df
        # that we are actually using
        given_columns.update(feature_columns)

        df_column_order = [c for c in df.columns if c in given_columns]

        if discrete_columns is None:
            discrete_column_set = set()
        else:
            discrete_column_set = set(discrete_columns)

        # Check for string columns and ensure they are considered discrete.
        for column_name in df.columns:
            # Check all columns being used, except time_column and
            # example_id_column which are not directly modeled.
            if (
                df[column_name].dtype == "O"
                and column_name in given_columns
                and column_name != time_column
                and column_name != example_id_column
            ):
                logging.info(
                    f"Marking column {column_name} as discrete because its type is string/object."
                )
                discrete_column_set.add(column_name)

        attribute_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in attribute_columns
        ]
        feature_types = [
            OutputType.DISCRETE if c in discrete_column_set else OutputType.CONTINUOUS
            for c in feature_columns
        ]

        if time_column:
            if example_id_column:
                # Assume all examples are for the same time points, e.g., always
                # from 2020 even if df has examples from different years.
                df_time_example = df[[time_column, example_id_column]]
                time_values = df_time_example.groupby(example_id_column).apply(
                    pd.DataFrame.to_numpy
                )[0][:, 0]

                time_column_values = list(sorted(time_values))
            else:
                time_column_values = list(sorted(df[time_column]))
        else:
            time_column_values = None

        return cls(
            attribute_columns=attribute_columns,
            feature_columns=feature_columns,
            example_id_column=example_id_column,
            time_column=time_column,
            discrete_columns=sorted(discrete_column_set),
            df_column_order=df_column_order,
            attribute_types=attribute_types,
            feature_types=feature_types,
            time_column_values=time_column_values,
        )

    @property
    def name(self) -> str:
        return "LongDataFrameConverter"

    @property
    def attribute_types(self):
        return self._attribute_types

    @property
    def feature_types(self):
        return self._feature_types

    def convert(self, df: pd.DataFrame) -> AttributeFeaturePair:

        if self._time_column is not None:
            sorted_df = df.sort_values(by=[self._time_column])
        else:
            sorted_df = df

        if self._example_id_column is not None:
            # Use example_id_column to split into separate time series
            df_features = sorted_df[self._feature_columns]

            features = np.stack(
                list(
                    df_features.groupby(sorted_df[self._example_id_column]).apply(
                        pd.DataFrame.to_numpy
                    )
                ),
                axis=0,
            )

            if self._attribute_columns:
                df_attributes = sorted_df[
                    self._attribute_columns + [self._example_id_column]
                ]

                # Check that attributes are the same for all rows with the same
                # example id. Use custom min and max functions that ignore nans.
                # Using pandas min() and max() functions leads to errors when a
                # single example has a mix of string and nan values for an
                # attribute across different rows because str and float are not
                # comparable.
                def custom_min(a):
                    return min((x for x in a if x is not np.nan), default=np.nan)

                def custom_max(a):
                    return max((x for x in a if x is not np.nan), default=np.nan)

                attribute_mins = df_attributes.groupby(self._example_id_column).apply(
                    lambda frame: frame.apply(custom_min)
                )
                attribute_maxes = df_attributes.groupby(self._example_id_column).apply(
                    lambda frame: frame.apply(custom_max)
                )

                for column in self._attribute_columns:
                    # Use custom list comprehension for the comparison to allow
                    # nan attribute values (nans don't compare equal so any
                    # example with an attribute of nan would fail the min/max
                    # equality check).
                    comparison = [
                        x is np.nan if y is np.nan else x == y
                        for x, y in zip(attribute_mins[column], attribute_maxes[column])
                    ]
                    if not np.all(comparison):
                        raise ValueError(
                            f"Attribute {column} is not constant within each example."
                        )

                attributes = (
                    df_attributes.groupby(self._example_id_column).min().to_numpy()
                )
            else:
                attributes = None
        else:
            # No example_id column provided to create multiple examples, so we
            # create one example from all time points.
            features = np.expand_dims(
                sorted_df[self._feature_columns].to_numpy(), axis=0
            )

            # Check that attributes are the same for all rows (since they are
            # all implicitly in the same example)
            for column in self._attribute_columns:
                if sorted_df[column].nunique() != 1:
                    raise ValueError(
                        f"Attribute {column} is not constant for all rows."
                    )

            if self._attribute_columns:
                # With one example, attributes should all be constant, so grab from
                # the first row. Need to add first (example) dimension.
                attributes = np.expand_dims(
                    sorted_df[self._attribute_columns].iloc[0, :].to_numpy(), axis=0
                )
            else:
                attributes = None

        return attributes, features

    def invert(
        self, attributes: Optional[np.ndarray], features: np.ndarray
    ) -> pd.DataFrame:
        num_examples = features.shape[0]
        num_time_points = features.shape[1]
        num_features = features.shape[2]

        if num_features != len(self._feature_columns):
            raise ValueError(
                "Unable to invert features back to data frame, "
                + f"converter expected {len(self._feature_columns)} features, "
                + f"received numpy array with {features.shape[2]}"
            )

        # Reshape so each time point is its own row in a 2d array
        long_features = features.reshape(-1, num_features)

        if self._attribute_columns:
            if attributes is None:
                raise ValueError(
                    "Data converter with attribute columns expects attributes array, received None"
                )
            # Repeat attribute rows for every time point in each example
            long_attributes = np.repeat(attributes, num_time_points, axis=0)

            df = pd.DataFrame(
                np.hstack((long_attributes, long_features)),
                columns=self._attribute_columns + self._feature_columns,
            )
        else:
            df = pd.DataFrame(
                long_features,
                columns=self._feature_columns,
            )

        # Convert discrete columns to int where possible.
        for c in self._discrete_columns:
            try:
                df[c] = df[c].astype("int")
            except ValueError:
                pass

        if self._example_id_column:
            # Use [0,1,2,...] for example_id
            # This may not match the style of the originally converted data
            df[self._example_id_column] = np.repeat(
                range(num_examples), num_time_points
            )

        if self._time_column:
            if self._time_column_values is None:
                raise RuntimeError("time_column is present, but not time_column_values")

            df[self._time_column] = np.tile(self._time_column_values, num_examples)

        return df[self._df_column_order]

    def _state_dict(self) -> Dict:
        return {
            "attribute_columns": self._attribute_columns,
            "feature_columns": self._feature_columns,
            "example_id_column": self._example_id_column,
            "time_column": self._time_column,
            "df_column_order": self._df_column_order,
            "discrete_columns": self._discrete_columns,
            "attribute_types": self._attribute_types,
            "feature_types": self._feature_types,
            "time_column_values": self._time_column_values,
        }


CONVERTER_CLASS_MAP = {
    "WideDataFrameConverter": _WideDataFrameConverter,
    "LongDataFrameConverter": _LongDataFrameConverter,
}


def find_max_consecutive_nans(array: np.array) -> int:
    """
    Returns the maximum number of consecutive NaNs in an array.

    Args:
        array: 1-d numpy array of time series per example.

    Returns:
        max_cons_nan: The maximum number of consecutive NaNs in a times series array.

    """
    # The number of consecutive nans are listed based on the index difference between the non-null values.
    max_cons_nan = np.max(
        np.diff(np.concatenate(([-1], np.where(~np.isnan(array))[0], [len(array)]))) - 1
    )
    return max_cons_nan


def validation_check(
    array: np.ndarray,
    invalid_examples_ratio_cutoff: float = 0.5,
    nans_ratio_cutoff: float = 0.1,
    consecutive_nans_max: int = 5,
    consecutive_nans_ratio_cutoff: float = 0.05,
) -> np.array:

    """Checks if continuous features of examples are valid.

    Returns a 1-d numpy array of booleans with shape (#examples) indicating
    valid examples.
    Examples with continuous features fall into 3 categories: good, valid (fixable) and
    invalid (non-fixable).
    - "Good" examples have no NaNs.
    - "Valid" examples have a low percentage of nans and a below a threshold number of
    consecutive NaNs.
    - "Invalid" are the rest, and are marked "False" in the returned array.  Later on,
    these are omitted from training. If there are too many, later, we error out.

    Args:
        array: 3-d numpy array of continuous features with
        shape (#examples,max_sequence_length, #continuous features).
        invalid_examples_ratio_cutoff: Error out if the invalid examples ratio in the dataset
        is higher than this value.
        nans_ratio_cutoff: If the percentage of nans for any continuous feature in an example
        is greater than this value, the example is invalid.
        consecutive_nans_max: If the maximum number of consecutive nans in a continuous
        feature is greater than this number, then that example is invalid.
        consecutive_nans_ratio_cutoff: If the maximum number of consecutive nans in a
        continuous feature is greater than this ratio times the length of the example
        (number samples), then the example is invalid.

    Returns:
        valid_examples : 1-d numpy array of booleans indicating valid examples with
        shape (#examples).

    """
    # Check for the nans ratio per examples and feature.
    # nan_ratio_feature is a 2-d numpy array of size (#examples,#features)

    nan_ratio_feature = np.mean(np.isnan(array), axis=1)
    nan_ratio = nan_ratio_feature < nans_ratio_cutoff

    # Check for max number of consecutive NaN values per example and feature.
    # cons_nans_feature is a 2-d numpy array of size (#examples,#features)
    cons_nans_feature = np.apply_along_axis(find_max_consecutive_nans, 1, array)
    cons_nans = cons_nans_feature < min(
        consecutive_nans_max,
        max(2, int(consecutive_nans_ratio_cutoff * array.shape[1])),
    )

    # The two above checks should pass for a valid example for all features, otherwise
    #  the example is invalid.
    valid_examples_per_feature = np.logical_and(nan_ratio, cons_nans)
    valid_examples = np.all(valid_examples_per_feature, axis=1)

    if np.mean(valid_examples) < invalid_examples_ratio_cutoff:
        raise ValueError(
            f"More than {100*invalid_examples_ratio_cutoff}% invalid examples in the continuous features. Please reduce the ratio of the NaNs and try again!"
        )

    if (~valid_examples).any():
        logger.warning(
            f"There are {sum(~valid_examples)} examples that have too many nan values in numeric features, accounting for {np.mean(~valid_examples)*100}% of all examples. These invalid examples will be omitted from training.",
            extra={"user_log": True},
        )

    return valid_examples


def nan_linear_interpolation(arrays: np.ndarray) -> np.ndarray:
    """Replaces all NaNs via linear interpolation.

    Args:
        arrays: 3-d numpy array of continuous features, with shape
        (#examples, max_sequence_length, #continuous features)

    Returns:
        arrays: 3-d numpy array where NaNs are replaced via
        linear interpolation.

    """
    examples = arrays.shape[0]
    features = arrays.shape[2]

    for exp in range(examples):
        for f in range(features):
            array = arrays[exp, :, f]
            if np.isnan(array).any():
                nans = np.isnan(array)
                ind_func = lambda z: z.nonzero()[0]
                array[nans] = np.interp(ind_func(nans), ind_func(~nans), array[~nans])

    return arrays
