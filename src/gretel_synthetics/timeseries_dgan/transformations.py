"""Module for converting data to and from internal DGAN representation."""

import abc
import uuid

from typing import cast, List, Optional, Tuple, Union

import numpy as np

from category_encoders import BinaryEncoder, OneHotEncoder
from gretel_synthetics.timeseries_dgan.config import Normalization, OutputType
from scipy.stats import mode


def _new_uuid() -> str:
    """Return a random uuid prefixed with 'gretel-'."""
    return f"gretel-{uuid.uuid4().hex}"


class Output(abc.ABC):
    """Stores metadata for a variable, used for both features and attributes."""

    def __init__(self, name: str):
        self.name = name

        self.is_fit = False

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of the transformed data produced for this variable."""
        ...

    def fit(self, column: np.ndarray):
        """Fit metadata and encoder params to data.

        Args:
            column: 1-d numpy array
        """
        if len(column.shape) != 1:
            raise ValueError("Expected 1-d numpy array for fit()")

        self._fit(column)
        self.is_fit = True

    def transform(self, column: np.ndarray) -> np.ndarray:
        """Transform data to internal representation.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array
        """
        if len(column.shape) != 1:
            raise ValueError("Expected 1-d numpy array for transform()")

        if not self.is_fit:
            raise RuntimeError("Cannot transform before output is fit()")
        else:
            return self._transform(column)

    def inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Inverse transform from internal representation to original data space.

        Args:
            columns: 2-d numpy array

        Returns:
            1-d numpy array in original data space
        """
        if not self.is_fit:
            raise RuntimeError("Cannot inverse transform before output is fit()")
        else:
            return self._inverse_transform(columns)

    @abc.abstractmethod
    def _fit(self, column: np.ndarray):
        ...

    @abc.abstractmethod
    def _transform(self, columns: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        ...


class OneHotEncodedOutput(Output):
    """Metadata for a one-hot encoded variable."""

    def __init__(self, name: str, dim=None):
        """
        Args:
            name: name of variable
            dim: use to directly setup encoder for [0,1,2,,...,dim-1] values, if
                not None, calling fit() is not required. Provided for easier
                backwards compatability. Preferred usage is dim=None and then
                call fit() on the instance.
        """
        super().__init__(name)

        if dim is not None:
            self.fit(np.arange(dim))

    @property
    def dim(self) -> int:
        """Dimension of the transformed data produced by one-hot encoding."""
        if self.is_fit:
            return len(self._encoder.get_feature_names())
        else:
            raise RuntimeError("Cannot return dim before output is fit()")

    def _fit(self, column: np.ndarray):
        """Fit one-hot encoder.

        Args:
            column: 1-d numpy array
        """
        # Use cols=0 to always do the encoding, even if the input is integer or
        # float.
        self._encoder = OneHotEncoder(cols=0, return_df=False)

        self._encoder.fit(column)

    def _transform(self, column: np.ndarray) -> np.ndarray:
        """Apply one-hot encoding.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array of encoded data
        """
        return self._encoder.transform(column).astype("float", casting="safe")

    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Invert one-hot encoding.

        Args:
            columns: 2-d numpy array of floats or integers

        Returns:
            1-d numpy array
        """
        if len(columns.shape) != 2:
            raise ValueError(
                f"Expected 2-d numpy array, received shape={columns.shape}"
            )
        # Category encoders only inverts exact match binary rows, so need to do
        # argmax and then convert back to full binary matrix.
        # Might be more efficient to eventually do everything ourselves and not
        # use OneHotEncoder.
        indices = np.argmax(columns, axis=1)
        b = np.zeros(columns.shape)
        b[np.arange(len(indices)), indices] = 1

        return self._encoder.inverse_transform(b).flatten()


class BinaryEncodedOutput(Output):
    """Metadata for a binary encoded variable."""

    def __init__(self, name: str, dim=None):
        """
        Args:
            name: name of variable
            dim: use to directly setup encoder for [0,1,2,,...,dim-1] values, if
                not None, calling fit() is not required. Provided for easier
                backwards compatability. Preferred usage is dim=None and then
                call fit() on the instance.
        """
        super().__init__(name)

        self._convert_to_int = False

        if dim is not None:
            self.fit(np.arange(dim))

    @property
    def dim(self) -> int:
        """Dimension of the transformed data produced by binary encoding."""
        if self.is_fit:
            return len(self._encoder.get_feature_names())
        else:
            raise RuntimeError("Cannot return dim before output is fit()")

    def _fit(self, column: np.ndarray):
        """Fit binary encoder.


        Args:
            column: 1-d numpy array
        """
        # Use cols=0 to always do the encoding, even if the input is integer or
        # float.
        self._encoder = BinaryEncoder(cols=0, return_df=False)

        if type(column) != np.array:
            column = np.array(column)
        else:
            column = column.copy()

        # BinaryEncoder fails a lot if the input is integer (tries to cast to
        # int during inverse transform, but often have NaNs). So force any
        # numeric column to float.
        if np.issubdtype(column.dtype, np.integer):
            column = column.astype("float")
            self._convert_to_int = True

        # Use proxy value for nans if present so we can decode them explicitly
        # and differentiate from decoding failures.
        nan_mask = [x is np.nan for x in column]
        if np.sum(nan_mask) > 0:
            self._nan_proxy = _new_uuid()
            # Always make a copy at beginning of this function, so in place
            # change is okay.
            column[nan_mask] = self._nan_proxy
        else:
            self._nan_proxy = None

        # Store mode to use for unmapped binary codes.
        self._mode = mode(column).mode[0]

        self._encoder.fit(column)

    def _transform(self, column: np.ndarray) -> np.ndarray:
        """Apply binary encoding.

        Args:
            column: 1-d numpy array

        Returns:
            2-d numpy array of encoded data
        """
        column = column.copy()
        if self._nan_proxy:
            nan_mask = [x is np.nan for x in column]
            column[nan_mask] = self._nan_proxy

        return self._encoder.transform(column).astype("float", casting="safe")

    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Invert binary encoding.

        Args:
            columns: 2-d numpy array of floats or integers

        Returns:
            1-d numpy array
        """
        if len(columns.shape) != 2:
            raise ValueError(
                f"Expected 2-d numpy array, received shape={columns.shape}"
            )

        # Threshold to binary matrix
        binary = (columns > 0.5).astype("int")

        original_data = self._encoder.inverse_transform(binary).flatten()

        nan_mask = [x is np.nan for x in original_data]

        original_data[nan_mask] = self._mode

        # Now that decoding failure nans are replaced with the mode, replace
        # nan_proxy values with nans.
        if self._nan_proxy:
            nan_proxy_mask = [x == self._nan_proxy for x in original_data]
            original_data[nan_proxy_mask] = np.nan

        if self._convert_to_int:
            # TODO: store original type for conversion?
            original_data = original_data.astype("int")

        return original_data


class ContinuousOutput(Output):
    """Metadata for continuous variables."""

    def __init__(
        self,
        name: str,
        normalization: Normalization,
        apply_feature_scaling: bool,
        apply_example_scaling: bool,
        *,
        global_min: Optional[float] = None,
        global_max: Optional[float] = None,
    ):
        """
        Args:
            name: name of variable
            normalization: range of transformed value
            apply_feature_scaling: should values be scaled
            apply_example_scaling: should per-example scaling be used
            global_min: backwards compatability to set range in constructor,
                preferred to use fit()
            global_max: backwards compatability to set range in constructor
        """
        super().__init__(name)

        self.normalization = normalization

        self.apply_feature_scaling = apply_feature_scaling
        self.apply_example_scaling = apply_example_scaling

        if (global_min is None) != (global_max is None):
            raise ValueError("Must provide both global_min and global_max")

        if global_min is not None:
            self.is_fit = True
            self.global_min = global_min
            self.global_max = global_max

    @property
    def dim(self) -> int:
        """Dimension of transformed data."""
        return 1

    def _fit(self, column):
        """Fit continuous variable encoding/scaling.

        Args:
            column: 1-d numpy array
        """
        column = column.astype("float")
        self.global_min = np.nanmin(column)
        self.global_max = np.nanmax(column)

    def _transform(self, column: np.ndarray) -> np.ndarray:
        """Apply continuous variable encoding/scaling.

        Args:
            column: numpy array

        Returns:
            numpy array of rescaled data
        """
        column = column.astype("float")
        if self.apply_feature_scaling:
            return rescale(column, self.normalization, self.global_min, self.global_max)
        else:
            return column

    def _inverse_transform(self, columns: np.ndarray) -> np.ndarray:
        """Invert continus variable encoding/scaling.

        Args:
            columns: numpy array

        Returns:
            numpy array
        """
        if self.apply_feature_scaling:
            return rescale_inverse(
                columns, self.normalization, self.global_min, self.global_max
            )
        else:
            return columns


def create_outputs_from_data(
    attributes: Optional[np.ndarray],
    features: np.ndarray,
    attribute_types: Optional[List[OutputType]],
    feature_types: Optional[List[OutputType]],
    normalization: Normalization,
    apply_feature_scaling: bool = False,
    apply_example_scaling: bool = False,
    binary_encoder_cutoff: int = 150,
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
        binary_encoder_cutoff: use binary encoder (instead of one hot encoder) for
            any column with more than this many unique values
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
                binary_encoder_cutoff=binary_encoder_cutoff,
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
            binary_encoder_cutoff=binary_encoder_cutoff,
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
    binary_encoder_cutoff: int,
) -> Output:
    """Create a single output from data.

    Args:
        index: index of variable within attributes or features
        t: type of output
        data: numpy array of data just for this variable
        normalization: see documentation in create_outputs_from_data
        apply_feature_scaling: see documentation in create_outputs_from_data
        apply_example_scaling: see documentation in create_outputs_from_data
        binary_encoder_cutoff: see documentation in create_outputs_from_data

    Returns:
        Output metadata instance
    """
    if t == OutputType.CONTINUOUS:
        output = ContinuousOutput(
            name="a" + str(index),
            normalization=normalization,
            apply_feature_scaling=apply_feature_scaling,
            apply_example_scaling=apply_example_scaling,
        )

    elif t == OutputType.DISCRETE:
        if data.dtype == "float":
            unique_count = len(np.unique(data))
        else:
            # Convert to str to ensure all elements are comparable (so unique
            # works as expected). In particular, this converts nan to "nan"
            # which is comparable.
            unique_count = len(np.unique(data.astype("str")))

        if unique_count > binary_encoder_cutoff:
            output = BinaryEncodedOutput(name="a" + str(index))
        else:
            output = OneHotEncodedOutput(name="a" + str(index))

    else:
        raise RuntimeError(f"Unknown output type={t}")

    output.fit(data.flatten())

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
        if "OneHotEncodedOutput" in str(
            output.__class__
        ) or "BinaryEncodedOutput" in str(output.__class__):

            if variable_dim_index == 1:
                original_column = original_data[:, index]
                target_shape = (original_data.shape[0], -1)
            elif variable_dim_index == 2:
                original_column = original_data[:, :, index]
                target_shape = (original_data.shape[0], original_data.shape[1], -1)
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            transformed_data = output.transform(original_column.flatten())

            parts.append(transformed_data.reshape(target_shape))

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

            feature_scaled = output.transform(raw.flatten()).reshape(raw.shape)

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
            np.concatenate(parts, axis=variable_dim_index, dtype="float"),
            np.concatenate(additional_attribute_parts, axis=1, dtype="float"),
        )
    else:
        return np.concatenate(parts, axis=variable_dim_index, dtype="float")


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
        if "OneHotEncodedOutput" in str(
            output.__class__
        ) or "BinaryEncodedOutput" in str(output.__class__):

            if variable_dim_index == 1:
                v = transformed_data[
                    :, transformed_index : (transformed_index + output.dim)
                ]
                target_shape = (transformed_data.shape[0], 1)
            elif variable_dim_index == 2:
                v = transformed_data[
                    :, :, transformed_index : (transformed_index + output.dim)
                ]
                target_shape = (transformed_data.shape[0], transformed_data.shape[1], 1)
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )

            original = output.inverse_transform(v.reshape((-1, v.shape[-1])))

            parts.append(original.reshape(target_shape))
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

            original = output.inverse_transform(example_scaled)

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
                        apply_feature_scaling=False,
                        apply_example_scaling=False,
                        # TODO: are min/max really needed here since we aren't
                        # doing any scaling, could add an IdentityOutput instead?
                        global_min=(
                            0.0
                            if output.normalization == Normalization.ZERO_ONE
                            else -1.0
                        ),
                        global_max=1.0,
                    )
                )
                # The half-range variable always uses ZERO_ONE normalization
                # because it should always be positive.
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_half_range",
                        normalization=Normalization.ZERO_ONE,
                        apply_feature_scaling=False,
                        apply_example_scaling=False,
                        global_min=0.0,
                        global_max=1.0,
                    )
                )

    return additional_attribute_outputs
