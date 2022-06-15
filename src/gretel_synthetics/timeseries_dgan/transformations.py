"""Module for converting data to and from internal DGAN representation."""

from dataclasses import dataclass, field
from typing import cast, List, Optional, Tuple, Union

import numpy as np

from gretel_synthetics.timeseries_dgan.config import Normalization, OutputType


@dataclass(frozen=True)
class Output:
    """Stores metadata for a variable, used for both features and attributes."""

    name: str
    dim: int


@dataclass(frozen=True)
class DiscreteOutput(Output):
    """Discrete (categorical) variable metadata."""

    pass


@dataclass(frozen=True)
class ContinuousOutput(Output):
    """Continuous variable metadata."""

    # All continuous outputs have dim=1
    dim: int = field(init=False, default=1)
    normalization: Normalization
    global_min: float
    global_max: float
    apply_feature_scaling: bool
    apply_example_scaling: bool


def create_outputs_from_data(
    attributes: Optional[np.ndarray],
    features: np.ndarray,
    attribute_types: Optional[List[OutputType]],
    feature_types: Optional[List[OutputType]],
    normalization: Normalization,
    apply_feature_scaling: bool = False,
    apply_example_scaling: bool = False,
) -> Tuple[Optional[List[Output]], List[Output]]:
    """Create output metadata from data.

    Args:
        attributes: 2d numpy array of attributes
        features: 3d numpy array of features
        attribute_types: variable type for each attribute, assumes continuous if None
        feature_types: variable type for each feature, assumes continuous if None
        normalization: internal representation for continuous variables, scale
            to [0,1] or [-1,1]
        apply_feature_scaling: scale continuous variables inside the model, if
            False inputs must already be scaled to [0,1] or [-1,1]
        apply_example_scaling: include midpoint and half-range as additional
            attributes for each feature and scale per example, improves
            performance when time series ranges are highly variable
    """
    attribute_outputs = None
    if attributes is not None:
        if attribute_types is None:
            attribute_types = [OutputType.CONTINUOUS] * attributes.shape[1]
        elif len(attribute_types) != attributes.shape[1]:
            raise RuntimeError(
                "attribute_types must be the same length as the 2nd (last) dimension of attributes"
            )
        attribute_types = cast(List[OutputType], attribute_types)
        attribute_outputs = [
            create_output(
                index,
                t,
                attributes[:, index],
                normalization=normalization,
                apply_feature_scaling=apply_feature_scaling,
                # Attributes can never be normalized per example since there's
                # only 1 value for each variable per example.
                apply_example_scaling=False,
            )
            for index, t in enumerate(attribute_types)
        ]

    if feature_types is None:
        feature_types = [OutputType.CONTINUOUS] * features.shape[2]
    elif len(feature_types) != features.shape[2]:
        raise RuntimeError(
            "feature_types must be the same length as the 3rd (last) dimemnsion of features"
        )
    feature_types = cast(List[OutputType], feature_types)

    feature_outputs = [
        create_output(
            index,
            t,
            features[:, :, index],
            normalization=normalization,
            apply_feature_scaling=apply_feature_scaling,
            apply_example_scaling=apply_example_scaling,
        )
        for index, t in enumerate(feature_types)
    ]

    return attribute_outputs, feature_outputs


def create_output(
    index: int,
    t: OutputType,
    data: np.ndarray,
    normalization: Normalization,
    apply_feature_scaling: bool,
    apply_example_scaling: bool,
) -> Output:
    """Create a single output from data.

    Args:
        index: index of variable within attributes or features
        t: type of output
        data: numpy array of data just for this variable
        normalization: see documentation in create_outputs_from_data
        apply_feature_scaling: see documentation in create_outputs_from_data
        apply_example_scaling: see documentation in create_outputs_from_data

    Returns:
        Output metadata instance
    """
    if t == OutputType.CONTINUOUS:
        output = ContinuousOutput(
            name="a" + str(index),
            normalization=normalization,
            global_min=np.min(data),
            global_max=np.max(data),
            apply_feature_scaling=apply_feature_scaling,
            apply_example_scaling=apply_example_scaling,
        )
    elif t == OutputType.DISCRETE:
        output = DiscreteOutput(
            name="a" + str(index),
            dim=1 + np.int32(np.max(data)),
        )
    else:
        raise RuntimeError(f"Unknown output type={t}")
    return output


def rescale(
    original: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    """Scale continuous variable to [0,1] or [-1,1].

    Args:
        original: data in original space
        normalization: output range for scaling, ZERO_ONE or MINUSONE_ONE
        global_min: minimum to use for scaling, either a scalar or has same
            shape as original
        global_max: maximum to use for scaling, either a scalar or has same
            shape as original

    Returns:
        Data in transformed space
    """

    range = np.maximum(global_max - global_min, 1e-6)
    if normalization == Normalization.ZERO_ONE:
        return (original - global_min) / range
    elif normalization == Normalization.MINUSONE_ONE:
        return (2.0 * (original - global_min) / range) - 1.0


def rescale_inverse(
    transformed: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    """Invert continuous scaling to map back to original space.

    Args:
        transformed: data in transformed space
        normalization: output range for scaling, ZERO_ONE or MINUSONE_ONE
        global_min: minimum to use for scaling, either a scalar or has same
            dimension as original.shape[0] for scaling each time series
            independently
        global_max: maximum to use for scaling, either a scalar or has same
            dimension as original.shape[0]

    Returns:
        Data in original space
    """
    range = global_max - global_min
    if normalization == Normalization.ZERO_ONE:
        return transformed * range + global_min
    elif normalization == Normalization.MINUSONE_ONE:
        return ((transformed + 1.0) / 2.0) * range + global_min


def transform(
    original_data: Optional[np.ndarray],
    outputs: List[Output],
    variable_dim_index: int,
    num_examples: Optional[int] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Transform data to internal representation expected by DoppelGANger.

    Specifically, performs the following changes:

    * Converts discrete variables to one-hot encoding
    * Scales continuous variables by feature or example min/max to [0,1] or
        [-1,1]
    * Create per example attributes with midpoint and half-range when
        apply_example_scaling is True

    Args:
        original_data: data to transform, 2d or 3d numpy array, or None
        outputs: Output metadata for each variable
        variable_dim_index: dimension of numpy array that contains the
            variables, for 2d numpy arrays this should be 1, for 3d should be 2
        num_examples: dimension of feature output array. If the original
            data is None, we want the empty/none torch array to match the first
            dimension of the feature output array. This makes sure that the
            TensorDataset module works smoothly. If the first dimensions are different,
            torch will give an error.

    Returns:
        Internal representation of data. A single numpy array if the input was a
        2d array or if no outputs have apply_example_scaling=True. A tuple of
        features, additional_attributes is returned when transforming features
        (a 3d numpy array) and example scaling is used. If the input data is
        None, then a single numpy array filled with nan's that has the first
        dimension shape of the number examples of the feature vector is
        returned.
    """
    additional_attribute_parts = []
    parts = []
    if original_data is None:
        return np.full((num_examples, 1), np.nan)

    for index, output in enumerate(outputs):
        # NOTE: isinstance(output, DiscreteOutput) does not work consistently
        #       with all import styles in jupyter notebooks, using string
        #       comparison instead.
        if "DiscreteOutput" in str(output.__class__):
            output = cast(DiscreteOutput, output)

            if variable_dim_index == 1:
                indices = original_data[:, index].astype(int)
            elif variable_dim_index == 2:
                indices = original_data[:, :, index].astype(int)
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            if variable_dim_index == 1:
                b = np.zeros((len(indices), output.dim))
                b[np.arange(len(indices)), indices] = 1
            elif variable_dim_index == 2:
                b = np.zeros((indices.shape[0], indices.shape[1], output.dim))
                # From https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
                def all_idx(idx, axis):
                    grid = np.ogrid[tuple(map(slice, idx.shape))]
                    grid.insert(axis, idx)
                    return tuple(grid)

                b[all_idx(indices, axis=2)] = 1

            parts.append(b)
        elif "ContinuousOutput" in str(output.__class__):
            output = cast(ContinuousOutput, output)

            if variable_dim_index == 1:
                raw = original_data[:, index]
            elif variable_dim_index == 2:
                raw = original_data[:, :, index]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            if output.apply_feature_scaling:
                feature_scaled = rescale(
                    raw, output.normalization, output.global_min, output.global_max
                )
            else:
                feature_scaled = raw

            if output.apply_example_scaling:
                if variable_dim_index != 2:
                    raise RuntimeError(
                        "apply_example_scaling only applies to features, that is when the data has 3 dimensions"
                    )

                mins = np.min(feature_scaled, axis=1)
                maxes = np.max(feature_scaled, axis=1)

                additional_attribute_parts.append(
                    ((mins + maxes) / 2).reshape(mins.shape[0], 1)
                )
                additional_attribute_parts.append(
                    ((maxes - mins) / 2).reshape(mins.shape[0], 1)
                )

                mins = np.broadcast_to(
                    mins.reshape(mins.shape[0], 1),
                    (mins.shape[0], feature_scaled.shape[1]),
                )
                maxes = np.broadcast_to(
                    maxes.reshape(maxes.shape[0], 1),
                    (mins.shape[0], feature_scaled.shape[1]),
                )

                scaled = rescale(feature_scaled, output.normalization, mins, maxes)
            else:
                scaled = feature_scaled

            if variable_dim_index == 1:
                scaled = scaled.reshape((original_data.shape[0], 1))
            elif variable_dim_index == 2:
                scaled = scaled.reshape(
                    (original_data.shape[0], original_data.shape[1], 1)
                )
            parts.append(scaled)
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    if additional_attribute_parts:
        return (
            np.concatenate(parts, axis=variable_dim_index),
            np.concatenate(additional_attribute_parts, axis=1),
        )
    else:
        return np.concatenate(parts, axis=variable_dim_index)


def inverse_transform(
    transformed_data: np.ndarray,
    outputs: List[Output],
    variable_dim_index: int,
    additional_attributes: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Invert transform to map back to original space.

    Args:
        transformed_data: internal representation data
        outputs: Output metadata for each variable
        variable_dim_index: dimension of numpy array that contains the
            variables, for 2d numpy arrays this should be 1, for 3d should be 2
        additional_attributes: midpoint and half-ranges for outputs with
            apply_example_scaling=True

    Returns:
        If the input data provided is a numpy array with no-nans, then a numpy array of
        data in original space is returned. If the input data is nan-filled, the function
        returns None.
    """
    parts = []
    transformed_index = 0
    additional_attribute_index = 0
    if np.isnan(transformed_data).any():
        return None
    for output in outputs:
        if "DiscreteOutput" in str(output.__class__):
            output = cast(DiscreteOutput, output)

            if variable_dim_index == 1:
                onehot = transformed_data[
                    :, transformed_index : (transformed_index + output.dim)
                ]
            elif variable_dim_index == 2:
                onehot = transformed_data[
                    :, :, transformed_index : (transformed_index + output.dim)
                ]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )
            indices = np.argmax(onehot, axis=variable_dim_index)

            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            indices = indices.reshape(target_shape)

            parts.append(indices)
            transformed_index += output.dim
        elif "ContinuousOutput" in str(output.__class__):
            output = cast(ContinuousOutput, output)

            if variable_dim_index == 1:
                transformed = transformed_data[:, transformed_index]
            elif variable_dim_index == 2:
                transformed = transformed_data[:, :, transformed_index]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            if output.apply_example_scaling:
                if variable_dim_index != 2:
                    raise RuntimeError(
                        "apply_example_scaling only applies to features where the data has 3 dimensions"
                    )

                if additional_attributes is None:
                    raise RuntimeError(
                        "Must provide additional_attributes if apply_example_scaling=True"
                    )

                midpoint = additional_attributes[:, additional_attribute_index]
                half_range = additional_attributes[:, additional_attribute_index + 1]
                additional_attribute_index += 2

                mins = midpoint - half_range
                maxes = midpoint + half_range
                mins = np.expand_dims(mins, 1)
                maxes = np.expand_dims(maxes, 1)

                example_scaled = rescale_inverse(
                    transformed,
                    normalization=output.normalization,
                    global_min=mins,
                    global_max=maxes,
                )
            else:
                example_scaled = transformed

            if output.apply_feature_scaling:
                original = rescale_inverse(
                    example_scaled,
                    output.normalization,
                    output.global_min,
                    output.global_max,
                )
            else:
                original = example_scaled

            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            original = original.reshape(target_shape)

            parts.append(original)
            transformed_index += 1
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    return np.concatenate(parts, axis=variable_dim_index)


def create_additional_attribute_outputs(feature_outputs: List[Output]) -> List[Output]:

    """Create outputs for midpoint and half ranges.

    Returns list of additional attribute metadata. For each feature with
    apply_example_scaling=True, adds 2 attributes, one for the midpoint of the
    sequence and one for the half range.

    Args:
        feature_outputs: output metadata for features

    Returns:
        List of Output instances for additional attributes
    """
    additional_attribute_outputs = []
    for output in feature_outputs:
        if "ContinuousOutput" in str(output.__class__):
            output = cast(ContinuousOutput, output)
            if output.apply_example_scaling:
                # Assumes feature data is already normalized to [0,1] or
                # [-1,1] according to output.normalization before the
                # per-example midpoint and half-range are calculated. So no
                # normalization is needed for these variables.
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_midpoint",
                        normalization=output.normalization,
                        global_min=(
                            0.0
                            if output.normalization == Normalization.ZERO_ONE
                            else -1.0
                        ),
                        global_max=1.0,
                        apply_feature_scaling=False,
                        apply_example_scaling=False,
                    )
                )
                # The half-range variable always uses ZERO_ONE normalization
                # because it should always be positive.
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_half_range",
                        normalization=Normalization.ZERO_ONE,
                        global_min=0.0,
                        global_max=1.0,
                        apply_feature_scaling=False,
                        apply_example_scaling=False,
                    )
                )

    return additional_attribute_outputs
