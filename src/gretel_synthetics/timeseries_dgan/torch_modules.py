"""Internal module with torch implementation details of DGAN."""

from collections import OrderedDict
from typing import cast, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from gretel_synthetics.timeseries_dgan.config import Normalization
from gretel_synthetics.timeseries_dgan.transformations import (
    BinaryEncodedOutput,
    ContinuousOutput,
    OneHotEncodedOutput,
    Output,
)


class Merger(torch.nn.Module):
    """Merge several torch layers with same inputs into one concatenated layer."""

    def __init__(
        self,
        modules: Union[torch.nn.ModuleList, Iterable[torch.nn.Module]],
        dim_index: int,
    ):
        """Create Merge module that concatenates layers.

        Args:
            modules: modules (layers) to merge
            dim_index: dim for the torch.cat operation, often the last dimension
                of the tensors involved
        """
        super(Merger, self).__init__()
        if isinstance(modules, torch.nn.ModuleList):
            self.layers = modules
        else:
            self.layers = torch.nn.ModuleList(modules)

        self.dim_index = dim_index

    def forward(self, input):
        """Apply module to input.

        Args:
            input: whatever the layers are expecting, usually a Tensor or tuple
                of Tensors

        Returns:
            Concatenation of outputs from layers.
        """
        return torch.cat([m(input) for m in self.layers], dim=self.dim_index)


class OutputDecoder(torch.nn.Module):
    """Decoder to produce continuous or discrete output values as needed."""

    def __init__(self, input_dim: int, outputs: List[Output], dim_index: int):
        """Create decoder to make final output for a variable in DGAN.

        Args:
            input_dim: dimension of input vector
            outputs: list of variable metadata objects to generate
            dim_index: dim for torch.cat operation, often the last dimension
                of the tensors involved
        """
        super(OutputDecoder, self).__init__()
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("OutputDecoder received no outputs")

        self.dim_index = dim_index
        self.generators = torch.nn.ModuleList()

        for output in outputs:
            if "OneHotEncodedOutput" in str(output.__class__):
                output = cast(OneHotEncodedOutput, output)
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(int(input_dim), int(output.dim)),
                                ),
                                ("softmax", torch.nn.Softmax(dim=int(dim_index))),
                            ]
                        )
                    )
                )
            elif "BinaryEncodedOutput" in str(output.__class__):
                output = cast(BinaryEncodedOutput, output)
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(int(input_dim), int(output.dim)),
                                ),
                                (
                                    "sigmoid",
                                    torch.nn.Sigmoid(),
                                ),
                            ]
                        )
                    )
                )
            elif "ContinuousOutput" in str(output.__class__):
                output = cast(ContinuousOutput, output)
                if output.normalization == Normalization.ZERO_ONE:
                    normalizer = torch.nn.Sigmoid()
                elif output.normalization == Normalization.MINUSONE_ONE:
                    normalizer = torch.nn.Tanh()
                else:
                    raise RuntimeError(
                        f"Unsupported normalization='{output.normalization}'"
                    )
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(int(input_dim), int(output.dim)),
                                ),
                                ("normalization", normalizer),
                            ]
                        )
                    )
                )
            else:
                raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    def forward(self, input):
        """Apply module to input.

        Args:
            input: tensor with last dim of size input_dim

        Returns:
            Generated variables packed into a single tensor (in same order as outputs).
        """
        outputs = [generator(input) for generator in self.generators]
        merged = torch.cat(outputs, dim=self.dim_index)
        return merged


class SelectLastCell(torch.nn.Module):
    """Select just the last layer's hidden output from LSTM module."""

    def forward(self, x):
        """Apply module to input.

        Args:
            x: tensor output from an LSTM layer

        Returns:
            Tensor of last layer hidden output.
        """
        out, _ = x
        return out


class Generator(torch.nn.Module):
    """Generator networks for attributes and features of DGAN model."""

    def __init__(
        self,
        attribute_outputs: Optional[List[Output]],
        additional_attribute_outputs: Optional[List[Output]],
        feature_outputs: List[Output],
        max_sequence_len: int,
        sample_len: int,
        attribute_noise_dim: Optional[int],
        feature_noise_dim: int,
        attribute_num_units: Optional[int],
        attribute_num_layers: Optional[int],
        feature_num_units: int,
        feature_num_layers: int,
    ):
        """Create generator network.

        Args:
            attribute_outputs: metadata objects for attribute variables to
                generate
            additional_attribute_outputs: metadata objects for additional
                attribute variables to generate
            feature_outputs: metadata objects for feature variables to generate
            max_sequence_len: length of feature time sequences
            sample_len: # of time points to generate from each LSTM cell
            attribute_noise_dim: size of noise vector for attribute GAN
            feature_noise_dim: size of noise vector for feature GAN
            attribute_num_units: # of units per layer in MLP used to generate
                attributes
            attribute_num_layers: # of layers in MLP used to generate attributes
            feature_num_units: # of units per layer in LSTM used to generate
                features
            feature_num_layers: # of layers in LSTM used to generate features
        """
        super(Generator, self).__init__()
        assert max_sequence_len % sample_len == 0

        self.sample_len = sample_len
        self.max_sequence_len = max_sequence_len
        self.attribute_gen, attribute_dim = self._make_attribute_generator(
            attribute_outputs,
            attribute_noise_dim,
            attribute_num_units,
            attribute_num_layers,
        )
        (
            self.additional_attribute_gen,
            additional_attribute_dim,
        ) = self._make_attribute_generator(
            additional_attribute_outputs,
            attribute_noise_dim + attribute_dim,
            attribute_num_units,
            attribute_num_layers,
        )
        self.feature_gen = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "lstm",
                        torch.nn.LSTM(
                            int(
                                attribute_dim
                                + additional_attribute_dim
                                + feature_noise_dim
                            ),
                            int(feature_num_units),
                            int(feature_num_layers),
                            batch_first=True,
                        ),
                    ),
                    ("selector", SelectLastCell()),
                    (
                        "merger",
                        Merger(
                            [
                                OutputDecoder(
                                    int(feature_num_units), feature_outputs, dim_index=2
                                )
                                for _ in range(self.sample_len)
                            ],
                            dim_index=2,
                        ),
                    ),
                ]
            )
        )

    def _make_attribute_generator(
        self, outputs: List[Output], input_dim: int, num_units: int, num_layers: int
    ) -> torch.nn.Sequential:
        """Helper function to create generator network for attributes.

        Used to build the generater for both the attribute and additional
        attribute generation. The output dimension of the newly built
        generator is also outputted. This is useful when passing these
        attributes into other generators.

        Args:
            outputs: metadata objects for variables
            input_dim: size of input vectors (usually random noise)
            num_units: # of units per layer in MLP
            num_layers: # of layers in MLP

        Returns:
            Feed-forward MLP to generate attributes, wrapped in a
            torch.nn.Sequential module.
            Attribute dimension for LSTM layer size in generator.
        """
        if not outputs:
            return None, 0
        seq = []
        last_dim = int(input_dim)
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(int(last_dim), int(num_units)))
            seq.append(torch.nn.ReLU())
            seq.append(torch.nn.BatchNorm1d(int(num_units)))
            last_dim = int(num_units)

        seq.append(OutputDecoder(int(last_dim), outputs, dim_index=1))
        attribute_dim = sum(output.dim for output in outputs)
        return torch.nn.Sequential(*seq), int(attribute_dim)

    def forward(
        self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply module to input.

        Args:
            attribute_noise: noise tensor for attributes, 2d tensor of (batch
                size, attribute_noise_dim) shape
            feature_noise: noise tensor for features, 3d tensor of (batch size,
                max_sequence_len, feature_noise_dim) shape

        Returns:
            Tuple of generated tensors with attributes (if present), additional_attributes
            (if present), and features. The tuple is structured as follows: (attributes,
            additional_attributes, features). If attributes and/or additional_attributes is not
            present, an empty nan-filled tensor will be returned in the tuple. The function
            will always return a 3-element tensor tuple.
        """

        # Attribute features exist

        empty_tensor = torch.Tensor(np.full((1, 1), np.nan))

        if self.attribute_gen is not None:
            attributes = self.attribute_gen(attribute_noise)

            if self.additional_attribute_gen:
                # detach() should be equivalent to stop_gradient used in tf1 code.
                attributes_no_gradient = attributes.detach()
                additional_attribute_gen_input = torch.cat(
                    (attributes_no_gradient, attribute_noise), dim=1
                )

                additional_attributes = self.additional_attribute_gen(
                    additional_attribute_gen_input
                )
                combined_attributes = torch.cat(
                    (attributes, additional_attributes), dim=1
                )
            else:
                additional_attributes = empty_tensor
                combined_attributes = attributes

            # Use detach() to stop gradient flow
            combined_attributes_no_gradient = combined_attributes.detach()

            reshaped_attributes = torch.reshape(
                combined_attributes_no_gradient, (combined_attributes.shape[0], 1, -1)
            )
            reshaped_attributes = reshaped_attributes.expand(
                -1, feature_noise.shape[1], -1
            )

            feature_gen_input = torch.cat((reshaped_attributes, feature_noise), 2)

            features = self.feature_gen(feature_gen_input)

            features = torch.reshape(
                features, (features.shape[0], self.max_sequence_len, -1)
            )
            return attributes, additional_attributes, features
        else:

            if self.additional_attribute_gen:
                additional_attributes = self.additional_attribute_gen(attribute_noise)
                combined_attributes_no_gradient = additional_attributes.detach()
                reshaped_attributes = torch.reshape(
                    combined_attributes_no_gradient,
                    (additional_attributes.shape[0], 1, -1),
                )
                reshaped_attributes = reshaped_attributes.expand(
                    -1, feature_noise.shape[1], -1
                )
                feature_gen_input = torch.cat((reshaped_attributes, feature_noise), 2)
                features = self.feature_gen(feature_gen_input)
                features = torch.reshape(
                    features, (features.shape[0], self.max_sequence_len, -1)
                )
                return empty_tensor, additional_attributes, features

            else:
                features = self.feature_gen(feature_noise)
                features = torch.reshape(
                    features, (features.shape[0], self.max_sequence_len, -1)
                )
                return empty_tensor, empty_tensor, features


class Discriminator(torch.nn.Module):
    """Discriminator network for DGAN model."""

    def __init__(self, input_dim: int, num_layers: int = 5, num_units: int = 200):
        """Create discriminator network.

        Args:
            input_dim: size of input to discriminator network
            num_layers: # of layers in MLP used for discriminator
            num_units: # of units per layer in MLP used for discriminator
        """
        super(Discriminator, self).__init__()

        seq = []
        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(int(last_dim), int(num_units)))
            seq.append(torch.nn.ReLU())
            last_dim = num_units

        seq.append(torch.nn.Linear(int(last_dim), 1))

        self.seq = torch.nn.Sequential(*seq)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply module to input.

        Args:
            input: input tensor of shape (batch size, input_dim)

        Returns:
            Discriminator output with shape (batch size, 1).
        """
        return self.seq(input)
