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

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType
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

NumpyArrayPair = Tuple[np.ndarray, np.ndarray]
NumpyArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]


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

        self.attribute_column_names = None
        self.feature_column_names = None

    def train_numpy(
        self,
        attributes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        attribute_types: Optional[List[OutputType]] = None,
        feature_types: Optional[List[OutputType]] = None,
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
            attributes: 2-d numpy array of attributes for the training examples,
                size is (# of training examples) X (# of attributes)
            features: 3-d numpy array of time series features for the training,
                size is (# of training examples) X max_sequence_len X (# of
                features)
            attribute_types: Specification of Discrete or Continuous
                type for each variable of the attributes. Discrete attributes
                should be 0-indexed (not one-hot encoded). If None, assume all
                attributes are continuous. Ignored if the model was already
                built, either by passing *output params at initialization or
                because train_* was called previously.
            feature_types: Specification of Discrete or Continuous type
                for each variable of the features. Discrete attributes should be
                0-indexed (not one-hot encoded). If None, assume all features
                are continuous. Ignored if the model was already built, either
                by passing *output params at initialization or because train_*
                was called previously.
        """

        if attributes.shape[0] != features.shape[0]:
            raise RuntimeError(
                "First dimension of attributes and features must be the same length, i.e., the number of training examples."
            )

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

        internal_attributes = transform(
            attributes,
            self.attribute_outputs,
            variable_dim_index=1,
        )

        if self.additional_attribute_outputs:
            # Use dataset with 3 tensors: attributes, additional_attributes,
            # features
            (
                internal_features,
                internal_additional_attributes,
            ) = transform(features, self.feature_outputs, variable_dim_index=2)

            dataset = TensorDataset(
                torch.Tensor(internal_attributes),
                torch.Tensor(internal_additional_attributes),
                torch.Tensor(internal_features),
            )
        else:
            # No additional attributes, so use just 2 tensors: attributes, features
            internal_features = transform(
                features, self.feature_outputs, variable_dim_index=2
            )

            dataset = TensorDataset(
                torch.Tensor(internal_attributes),
                torch.Tensor(internal_features),
            )

        self._train(dataset)

    def train_dataframe(
        self,
        df: pd.DataFrame,
        df_attribute_columns: List[Union[str, int]],
        df_feature_columns: Optional[List[Union[str, int]]] = None,
        attribute_types: Optional[List[OutputType]] = None,
        feature_types: Optional[List[OutputType]] = None,
    ):
        """Train DGAN model on data in pandas DataFrame.

        Training data is passed as a DataFrame in "wide" format, with one row
        representing one example with attribute columns and 1 column per time
        point in the time series.

        Args:
            df: DataFrame of training data in "wide" format
            df_attribute_columns: list of column names containing attributes
            df_feature_columns: list of column names containing features, if None
                all non-attribute columns are used, Default: None
            attribute_types: Specification of variable types, see train_numpy()
                documentation for details
            feature_types: Specification of variable types, see train_numpy()
                documentation for details
        """
        attributes, features = self._extract_from_dataframe(
            df, df_attribute_columns, df_feature_columns
        )

        self.train_numpy(
            attributes=attributes,
            features=features,
            attribute_types=attribute_types,
            feature_types=feature_types,
        )

    def generate_numpy(
        self,
        n: Optional[int] = None,
        attribute_noise: Optional[torch.Tensor] = None,
        feature_noise: Optional[torch.Tensor] = None,
    ) -> NumpyArrayPair:
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
                np.concatenate(d, axis=0) for d in zip(*internal_data_list)
            )

        else:
            if attribute_noise is None or feature_noise is None:
                raise RuntimeError(
                    "generate() must receive either n or both attribute_noise and feature_noise"
                )
            attribute_noise = attribute_noise.to(self.device)
            feature_noise = feature_noise.to(self.device)

            internal_data = self._generate(attribute_noise, feature_noise)

        if self.additional_attribute_outputs:
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
        else:
            internal_attributes, internal_features = internal_data

            attributes = inverse_transform(
                internal_attributes, self.attribute_outputs, variable_dim_index=1
            )
            features = inverse_transform(
                internal_features, self.feature_outputs, variable_dim_index=2
            )

        if n is not None:
            # Truncate to requested length
            attributes = attributes[:n]
            features = features[:n]

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
            pandas DataFrame in "wide" format
        """

        attributes, features = self.generate_numpy(n, attribute_noise, feature_noise)

        if features.shape[2] != 1:
            raise RuntimeError(
                "Generating a dataframe is not supported with more than 1 feature variable"
            )

        data = np.concatenate(
            (attributes, features.reshape(features.shape[0], features.shape[1])), axis=1
        )

        columns = None
        if (
            self.attribute_column_names is not None
            and self.feature_column_names is not None
        ):
            columns = np.concatenate(
                (self.attribute_column_names, self.feature_column_names),
            )

        df = pd.DataFrame(data, columns=columns)

        return df

    def _build(
        self,
        attribute_outputs: List[Output],
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
        self.generator.to(self.device)

        attribute_dim = sum(output.dim for output in attribute_outputs)
        additional_attribute_dim = 0
        if self.additional_attribute_outputs:
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
        self.feature_discriminator.to(self.device)

        self.attribute_discriminator = None
        if self.config.use_attribute_discriminator:
            self.attribute_discriminator = Discriminator(
                attribute_dim + additional_attribute_dim,
                num_layers=5,
                num_units=200,
            )
            self.attribute_discriminator.to(self.device)

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
        and wrapped in a torch Dataset.

        Args:
            dataset: torch Dataset containing tuple of (attributes, features)
                or (attributes, additional_attributes, features)
        """

        loader = DataLoader(
            dataset, self.config.batch_size, shuffle=True, drop_last=True
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

        for epoch in range(self.config.epochs):
            logger.info(f"epoch: {epoch}")

            for real_batch in loader:
                global_step += 1
                attribute_noise = self.attribute_noise_func(self.config.batch_size)
                feature_noise = self.feature_noise_func(self.config.batch_size)

                # Both real and generated batch are a tuple of tensors. If
                # self.additional_attribute_outputs is non-empty, they are
                # 3-element tuples with attribute, additional_attribute, and
                # feature tensors. Otherwise, no additional attributes are used
                # by the model and they are 2-element tuples with attribute and
                # feature tensors.
                generated_batch = self.generator(attribute_noise, feature_noise)

                real_batch = [x.to(self.device) for x in real_batch]

                for index, b in enumerate(generated_batch):
                    if torch.isnan(b).any():
                        logger.warning(f"found nans in generated_batch index={index}")

                for _ in range(self.config.discriminator_rounds):
                    opt_discriminator.zero_grad()
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

                    loss.backward(retain_graph=True)
                    opt_discriminator.step()

                    if opt_attribute_discriminator is not None:
                        opt_attribute_discriminator.zero_grad()
                        # Exclude features (last element of batches) for
                        # attribute discriminator
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

                        attribute_loss.backward(retain_graph=True)
                        opt_attribute_discriminator.step()

                for _ in range(self.config.generator_rounds):
                    opt_generator.zero_grad()
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

                    loss.backward()
                    opt_generator.step()

    def _generate(
        self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor
    ) -> Union[NumpyArrayPair, NumpyArrayTriple]:
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
        inputs = list(batch)
        # Flatten the features
        inputs[-1] = torch.reshape(inputs[-1], (inputs[-1].shape[0], -1))

        input = torch.cat(inputs, dim=1)

        output = self.feature_discriminator(input)
        return output

    def _discriminate_attributes(self, batch) -> torch.Tensor:
        """Internal helper function to apply the GAN attribute discriminator.

        Args:
            batch: tuple of internal data, either 1 element for attributes
                or 2 elements for attributes and additional_attributes

        Returns:
            Output for GAN attribute discriminator.
        """
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

    def _extract_from_dataframe(
        self,
        df: pd.DataFrame,
        attribute_columns: List[Union[str, int]],
        feature_columns: Optional[List[Union[str, int]]] = None,
    ) -> NumpyArrayPair:
        """Extract attribute and feature arrays from a single pandas DataFrame

        Note this method only supports time series of 1 variable where the time
        steps are represented as separate columns in the DataFrame.

        Args:

            df: DataFrame of time series data in a "wide" format, each row is an
                example with some columns as attributes and a column for each time
                step in the time series.
            attribute_columns: column names or indices for the attributes, may be an
                empty list if there are no attributes
            feature_columns: column names or indices for the features, if not
                specified, all non-attribute columns will be used

        Returns:
            Tuple of (attributes, features)
        """
        attributes = df[attribute_columns].to_numpy()

        if feature_columns is None:
            features_df = df.drop(columns=attribute_columns)
        else:
            features_df = df[feature_columns]

        # Store column names so we can make a similar dataframe when generating
        # synthetic data.
        self.attribute_column_names = attribute_columns
        self.feature_column_names = features_df.columns

        # Convert from 2-d to 3-d array with dimension of 1 on the last dim (the
        # number of variables at each time point).
        features = np.expand_dims(features_df.to_numpy(), axis=-1)

        return attributes, features

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

        if self.attribute_column_names is not None:
            state["attribute_column_names"] = self.attribute_column_names
            state["feature_column_names"] = self.feature_column_names

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

        if "attribute_column_names" in state and "feature_column_names" in state:
            dgan.attribute_column_names = state["attribute_column_names"]
            dgan.feature_column_names = state["feature_column_names"]

        return dgan
