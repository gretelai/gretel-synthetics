import logging

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from packaging import version
from torch import optim
from torch.nn import (
    BatchNorm1d,
    Dropout,
    functional,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
)

from gretel_synthetics.actgan.base import BaseSynthesizer, random_state
from gretel_synthetics.actgan.column_encodings import (
    BinaryColumnEncoding,
    FloatColumnEncoding,
    OneHotColumnEncoding,
)
from gretel_synthetics.actgan.data_sampler import DataSampler
from gretel_synthetics.actgan.data_transformer import DataTransformer
from gretel_synthetics.actgan.structures import (
    ColumnType,
    ConditionalVectorType,
    EpochInfo,
)
from gretel_synthetics.actgan.train_data import TrainData
from gretel_synthetics.typing import DFLike

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NOTE on data terminology used in ACTGAN: This module operates with 3 different
# representations of the training data (and generated synthetic data).
#
# - original - input data as received in API calls, as DataFrame, same format
#       and style is returned for synthetic samples
# - transformed - compact representation after applying DataTransformer (and
#       usually stored in TrainData instances), columns here are always numeric,
#       but may be in a more compact decoded form than what the actual DNN works
#       on, in particular one-hot or binary encoded columns are stored as
#       integer indices, instead of multiple columns, also known as decoded
# - encoded - representation passed directly to DNNs and should be in proper
#       float32 dtype
#
# During training we apply the transformations from original -> transformed ->
# encoded. And for generation the process reverses, going from encoded
# representation back to the original format.


class Discriminator(Module):
    """Discriminator for the ACTGANSynthesizer."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=10, lambda_=10
    ):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the ACTGANSynthesizer."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the ACTGANSynthesizer."""

    def __init__(self, embedding_dim: int, generator_dim: Sequence[int], data_dim: int):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


def _gumbel_softmax_stabilized(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
):
    """Deals with the instability of the gumbel_softmax for older versions of torch.
    For more details about the issue:
    https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
    Args:
        logits […, num_features]:
            Unnormalized log probabilities
        tau:
            Non-negative scalar temperature
        hard (bool):
            If True, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
        dim (int):
            A dimension along which softmax will be computed. Default: -1.
    Returns:
        Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
    """
    for i in range(10):
        transformed = functional.gumbel_softmax(
            logits, tau=tau, hard=hard, eps=eps, dim=dim
        )
        if not torch.isnan(transformed).any():
            return transformed
    raise ValueError("gumbel_softmax returning NaN.")


class ACTGANSynthesizer(BaseSynthesizer):
    """Anyway Conditional Table GAN Synthesizer.

    This is the core class of the ACTGAN interface.

    Args:
        embedding_dim:
            Size of the random sample passed to the Generator.
        generator_dim:
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided.
        discriminator_dim:
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided.
        generator_lr:
            Learning rate for the generator.
        generator_decay:
            Generator weight decay for the Adam Optimizer.
        discriminator_lr:
            Learning rate for the discriminator.
        discriminator_decay:
            Discriminator weight decay for the Adam Optimizer.
        batch_size:
            Number of data samples to process in each step.
        discriminator_steps:
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875.
        binary_encoder_cutoff:
            For any given column, the number of unique values that should exist before
            switching over to binary encoding instead of OHE. This will help reduce
            memory consumption for datasets with a lot of unique values.
        binary_encoder_nan_handler:
            Binary encoding currently may produce errant NaN values during reverse transformation. By default
            these NaN's will be left in place, however if this value is set to "mode" then those NaN's will
            be replaced by a random value that is a known mode for a given column.
        cbn_sample_size:
            Number of rows to sample from each column for identifying clusters for the cluster-based normalizer.
            This only applies to float columns. If set to ``0``, no sampling is done and all values are considered,
            which may be very slow.
        log_frequency:
            Whether to use log frequency of categorical levels in conditional
            sampling.
        verbose:
            Whether to have log progress results.
        epochs:
            Number of training epochs.
        epoch_callback:
            If set to a callable, call the function with `EpochInfo` as the arg
        pac:
            Number of samples to group together when applying the discriminator.
        cuda:
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
        conditional_vector_type:
            Type of conditional vector to include in input to the generator.
            Influences how effective and flexible the native conditional
            generation is. Options include SINGLE_DISCRETE (original CTGAN
            setup) and ANYWAY.
        conditional_select_mean_columns:
            Target number of columns to select for conditioning on average
            during training. Only used for ANYWAY conditioning. Use if typical
            number of columns to seed on is known. If set,
            conditional_select_column_prob must be None. Equivalent to setting
            conditional_select_column_prob to conditional_select_mean_columns /
            # of columns.
        conditional_select_column_prob:
            Probability to select any given column to be conditioned on during
            training. Only used for ANYWAY conditioning. If set,
            conditional_select_mean_columns must be None.
        reconstruction_loss_coef:
            Multiplier on reconstruction loss, higher values focus the generator
            optimization more on accurate conditional vector generation.
        force_conditioning:
            Directly set the requested conditional generation columns in
            generated data. Will bypass rejection sampling and be faster, but
            may reduce quality of the generated data and correlations between
            conditioned columns and other variables may be weaker.
    """

    def __init__(
        self,
        embedding_dim: int,
        generator_dim: Sequence[int],
        discriminator_dim: Sequence[int],
        generator_lr: float,
        generator_decay: float,
        discriminator_lr: float,
        discriminator_decay: float,
        batch_size: int,
        discriminator_steps: int,
        binary_encoder_cutoff: int,
        binary_encoder_nan_handler: Optional[str],
        cbn_sample_size: Optional[int],
        log_frequency: bool,
        verbose: bool,
        epochs: int,
        epoch_callback: Optional[Callable],
        pac: int,
        cuda: bool,
        conditional_vector_type: ConditionalVectorType,
        conditional_select_mean_columns: Optional[float],
        conditional_select_column_prob: Optional[float],
        reconstruction_loss_coef: float,
        force_conditioning: bool,
    ):
        if batch_size % 2 != 0:
            raise ValueError("`batch_size` must be divisible by 2")

        if batch_size % pac != 0:
            raise ValueError("`batch_size` must be divisible by `pac` (defaults to 10)")

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._binary_encoder_cutoff = binary_encoder_cutoff
        self._binary_encoder_nan_handler = binary_encoder_nan_handler
        self._log_frequency = log_frequency
        self._cbn_sample_size = cbn_sample_size
        self._verbose = verbose
        self._epochs = epochs
        self._epoch_callback = epoch_callback
        self.pac = pac
        self._conditional_vector_type = conditional_vector_type

        if (
            conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE
            and conditional_select_column_prob is None
            and conditional_select_mean_columns is None
        ):
            raise ValueError(
                "conditional_select_column_prob and conditional_select_mean_columns are both None, exactly one of them must be set for ANYWAY training"
            )
        if (
            conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE
            and conditional_select_column_prob is not None
            and conditional_select_mean_columns is not None
        ):
            raise ValueError(
                "conditional_select_column_prob and conditional_select_mean_columns are both set, exactly one of them must be set for ANYWAY training"
            )

        if conditional_select_column_prob is not None and (
            conditional_select_column_prob < 0.0 or conditional_select_column_prob > 1.0
        ):
            raise ValueError(
                "conditional_select_column_prob must be between 0.0 and 1.0"
            )

        if (
            conditional_select_mean_columns is not None
            and conditional_select_mean_columns < 0
        ):
            raise ValueError("conditional_select_mean_columns must be an integer >=0")

        self._conditional_select_mean_columns = conditional_select_mean_columns
        self._conditional_select_column_prob = conditional_select_column_prob
        self._reconstruction_loss_coef = reconstruction_loss_coef
        self._force_conditioning = force_conditioning

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._transformer = None
        self._condvec_sampler = None
        self._generator = None

        self._activation_fns: List[
            Tuple[int, int, Callable[[torch.Tensor], torch.Tensor]]
        ] = []
        self._cond_loss_col_ranges: List[Tuple[int, int, int, int]] = []

        if self._epoch_callback is not None and not callable(self._epoch_callback):
            raise ValueError("`epoch_callback` must be a callable or `None`")

    _gumbel_softmax = staticmethod(
        functional.gumbel_softmax
        if version.parse(torch.__version__) >= version.parse("1.2.0")
        else _gumbel_softmax_stabilized
    )

    def _make_noise(self) -> torch.Tensor:
        """Create new random noise tensors for a batch.

        Returns:
            Tensor of random noise used as (part of the) input to generator
            network. Shape is [batch_size, embedding_dim].
        """
        # NOTE: speedup may be possible if we can reuse the mean and std tensors
        # here across calls to _make_noise.
        mean = torch.zeros(
            (self._batch_size, self._embedding_dim),
            dtype=torch.float32,
            device=self._device,
        )
        std = mean + 1.0
        return torch.normal(mean, std)

    def _apply_generator(
        self, fakez: torch.Tensor, fake_cond_vec: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply generator network.

        Args:
            fakez: Random noise (z-vectors), shape is [batch_size,
                embedding_dim]
            fake_cond_vec: Optional conditional vectors to guide generation,
                shape is [batch_size, cond_vec_dim]

        Returns:
            Tuple of direct generator output, and output after applying
            activation functions. Shape of both tensor outputs is [batch_size,
            data_dim]
        """
        if fake_cond_vec is None:
            input = fakez
        else:
            input = torch.cat([fakez, fake_cond_vec], dim=1)

        fake = self._generator(input)
        fakeact = self._apply_activate(fake)
        return fake, fakeact

    def _apply_discriminator(
        self,
        encoded: torch.Tensor,
        cond_vec: Optional[torch.Tensor],
        discriminator: Discriminator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply discriminator network.

        Args:
            encoded: Tensor of data in encoded representation to evaluate.
            cond_vec: Optional conditional vector

        Returns:
            Tuple of full input to the discriminator network and the output.
        """
        if cond_vec is None:
            input = encoded
        else:
            input = torch.cat([encoded, cond_vec], dim=1)
        y = discriminator(input)
        return input, y

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = [
            activation_fn(data[:, st:ed])
            for st, ed, activation_fn in self._activation_fns
        ]

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = [
            functional.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction="none",
            )
            for st, ed, st_c, ed_c in self._cond_loss_col_ranges
        ]

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _column_loss(
        self,
        data: torch.Tensor,
        data_act: torch.Tensor,
        cond_vec: torch.Tensor,
        activation_fn,
    ) -> torch.Tensor:
        # TODO: probably better to use an enum indicator for the column type,
        # instead of assuming these function inequalities work and also map to
        # the expected column types.
        if activation_fn == torch.tanh:
            # Assumes tanh is only used for 1 column at a time, ie data.shape[1]==1
            return functional.mse_loss(data_act, cond_vec, reduction="none").flatten()
        elif activation_fn == torch.sigmoid:
            bce = functional.binary_cross_entropy_with_logits(
                data,
                cond_vec,
                reduction="none",
            )
            # bce is computed for each representation column, so shape is
            # [batch_size, # of bits needed to represent unique values]. All
            # other losses in this function return a 1-d tensor of shape
            # [batch_size], so we take the mean loss across the representions
            # columns to convert from [batch_size, k] to [batch_size] shape.
            return bce.mean(dim=1)
        else:
            return functional.cross_entropy(
                data, torch.argmax(cond_vec, dim=1), reduction="none"
            )

    def _anyway_cond_loss(
        self,
        data: torch.Tensor,
        data_act: torch.Tensor,
        cond_vec: torch.Tensor,
        column_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss of data compared to the conditional vector.

        The generator output before and after applying activations are used here
        since loss functions for different column types use the value before or
        after applying the activation functions. We pass both to avoid
        recomputing any tensors.

        Args:
            data: direct output of generator, before activations are applied,
                shape [batch_size, encoded_dim]
            data_act: output of generator after activations are applied,
                shape [batch_size, encoded_dim]
            cond_vec: conditional vector that generator output should match,
                shape is [batch_size, encoded_dim]
            column_mask: 0/1 mask of columns selected for conditioning, shape is
                [batch_size, encoded_dim]

        Returns:
            Reconstruction loss
        """
        loss = [
            self._column_loss(
                data[:, st:ed], data_act[:, st:ed], cond_vec[:, st:ed], activation_fn
            )
            for st, ed, activation_fn in self._activation_fns
        ]
        loss = torch.stack(loss, dim=1)
        return (loss * column_mask).sum() / data.size()[0]

    def _validate_discrete_columns(
        self, train_data: DFLike, discrete_columns: Sequence[str]
    ) -> None:
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data: Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns:
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError("``train_data`` should be either pd.DataFrame or np.array.")

        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    def _prepare_batch(
        self, data_sampler: DataSampler
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a random subset of training data for one batch.

        Also prepares other required Tensors such as conditional vectors, for
        generator and discriminator training.

        Args:
            data_sampler: DataSampler instance that performs sampling

        Returns:
            Tuple of:
              - torch.Tensor or None, fake conditional vector (part of input to
                generator)
              - torch.Tensor or None, real conditional vector associated with
                the encoded real sample returned
              - torch.Tensor or None, column mask indicating which columns (in
                transformed representation) are set in the fake conditional
                vector
              - torch.Tensor, encoded real sample
        """

        if self._conditional_vector_type == ConditionalVectorType.SINGLE_DISCRETE:
            # CTGAN style conditional vector selecting exactly 1 discrete
            # variable
            fake_cond_vec, fake_column_mask, col, opt = data_sampler.sample_condvec(
                self._batch_size
            )

            if fake_cond_vec is None:
                real_encoded = data_sampler.sample_data(self._batch_size, None, None)
                real_cond_vec = None
            else:
                fake_cond_vec = torch.from_numpy(fake_cond_vec).to(self._device)
                fake_column_mask = torch.from_numpy(fake_column_mask).to(self._device)

                perm = np.random.permutation(self._batch_size)
                real_encoded = data_sampler.sample_data(
                    self._batch_size, col[perm], opt[perm]
                )
                real_cond_vec = fake_cond_vec[perm]

            real_encoded = torch.from_numpy(real_encoded.astype("float32")).to(
                self._device
            )

            return (
                fake_cond_vec,
                real_cond_vec,
                fake_column_mask,
                real_encoded,
            )
        else:
            (
                fake_cond_vec,
                real_encoded,
                fake_column_mask,
            ) = data_sampler.sample_anyway_cond_vec(self._batch_size)

            perm = np.random.permutation(self._batch_size)
            real_cond_vec = fake_cond_vec[perm]
            real_encoded = real_encoded[perm]

            # Convert everything to tensors. Could be performance benefits by
            # using torch in the sample_anyway_cond_vec function, moving more
            # computation into the tensor graph and the GPU.
            fake_cond_vec = torch.from_numpy(fake_cond_vec.astype("float32")).to(
                self._device
            )
            real_cond_vec = torch.from_numpy(real_cond_vec.astype("float32")).to(
                self._device
            )
            fake_column_mask = torch.from_numpy(fake_column_mask.astype("float32")).to(
                self._device
            )
            real_encoded = torch.from_numpy(real_encoded.astype("float32")).to(
                self._device
            )

            return fake_cond_vec, real_cond_vec, fake_column_mask, real_encoded

    @random_state
    def fit(
        self, train_data: DFLike, discrete_columns: Optional[Sequence[str]] = None
    ) -> None:
        transformed_train_data = self._pre_fit_transform(
            train_data, discrete_columns=discrete_columns
        )
        self._actual_fit(transformed_train_data)

    def _pre_fit_transform(
        self, train_data: DFLike, discrete_columns: Optional[Sequence[str]] = None
    ) -> TrainData:
        if discrete_columns is None:
            discrete_columns = ()

        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer(
            binary_encoder_cutoff=self._binary_encoder_cutoff,
            binary_encoder_nan_handler=self._binary_encoder_nan_handler,
            cbn_sample_size=self._cbn_sample_size,
            verbose=self._verbose,
        )
        self._transformer.fit(train_data, discrete_columns)

        train_data_dec = self._transformer.transform_decoded(train_data)

        self._activation_fns = []
        self._cond_loss_col_ranges = []

        st = 0
        st_c = 0
        for column_info in train_data_dec.column_infos:
            for enc in column_info.encodings:
                ed = st + enc.encoded_dim
                if isinstance(enc, FloatColumnEncoding):
                    self._activation_fns.append((st, ed, torch.tanh))
                elif isinstance(enc, BinaryColumnEncoding):
                    self._activation_fns.append((st, ed, torch.sigmoid))
                elif isinstance(enc, OneHotColumnEncoding):
                    self._activation_fns.append(
                        (st, ed, lambda data: self._gumbel_softmax(data, tau=0.2))
                    )
                    if column_info.column_type == ColumnType.DISCRETE:
                        ed_c = st_c + enc.encoded_dim
                        self._cond_loss_col_ranges.append((st, ed, st_c, ed_c))
                        st_c = ed_c
                else:
                    raise ValueError(f"Unexpected column encoding {type(enc)}")

                st = ed

        return train_data_dec

    def _actual_fit(self, train_data: TrainData) -> None:
        """Fit the ACTGAN Synthesizer models to the training data.

        Args:
            train_data: training data as a TrainData instance
        """

        epochs = self._epochs
        column_prob = None
        if self._conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE:
            if self._conditional_select_mean_columns is None:
                column_prob = self._conditional_select_column_prob
            else:
                # Clip to ensure probability is between 0.0 and 1.0.
                column_prob = min(
                    1.0,
                    self._conditional_select_mean_columns
                    / len(train_data.columns_and_data),
                )
            if column_prob > 0.5:
                logger.warn(
                    "Column selection probability for ANYWAY training is "
                    f"{column_prob} > 0.5, recommended to keep below 0.5 to "
                    "ensure the model can utilize the GAN noise. Use smaller "
                    "conditional_select_mean_columns or "
                    "conditional_select_column_prob < 0.5."
                )

        data_sampler = DataSampler(
            train_data,
            self._log_frequency,
            self._conditional_vector_type,
            column_prob,
        )
        self._condvec_sampler = data_sampler.condvec_sampler
        self._cond_vec_dim = data_sampler.cond_vec_dim

        data_dim = train_data.encoded_dim

        self._generator = Generator(
            self._embedding_dim + data_sampler.cond_vec_dim,
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + data_sampler.cond_vec_dim,
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for _ in range(steps_per_epoch):
                for _ in range(self._discriminator_steps):
                    # Optimize discriminator
                    fakez = self._make_noise()
                    (
                        fake_cond_vec,
                        real_cond_vec,
                        fake_column_mask,
                        real_encoded,
                    ) = self._prepare_batch(data_sampler)

                    fake, fakeact = self._apply_generator(fakez, fake_cond_vec)

                    fake_cat, y_fake = self._apply_discriminator(
                        fakeact, fake_cond_vec, discriminator
                    )
                    real_cat, y_real = self._apply_discriminator(
                        real_encoded, real_cond_vec, discriminator
                    )

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # Optimize generator
                fakez = self._make_noise()
                (
                    fake_cond_vec,
                    real_cond_vec,
                    fake_column_mask,
                    # Real data is unused here, possible speedup if we skip
                    # creating this Tensor for CTGAN style conditional vectors
                    _,
                ) = self._prepare_batch(data_sampler)

                fake, fakeact = self._apply_generator(fakez, fake_cond_vec)
                fake_cat, y_fake = self._apply_discriminator(
                    fakeact, fake_cond_vec, discriminator
                )

                if (
                    self._conditional_vector_type
                    == ConditionalVectorType.SINGLE_DISCRETE
                ):
                    if fake_cond_vec is None:
                        loss_reconstruction = torch.tensor(
                            0.0, dtype=torch.float32, device=self._device
                        )
                    else:
                        loss_reconstruction = self._cond_loss(
                            fake, fake_cond_vec, fake_column_mask
                        )
                else:
                    loss_reconstruction = self._anyway_cond_loss(
                        fake, fakeact, fake_cond_vec, fake_column_mask
                    )

                loss_g = (
                    -torch.mean(y_fake)
                    + self._reconstruction_loss_coef * loss_reconstruction
                )

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            _epoch = i + 1
            _loss_g = round(float(loss_g.detach().cpu()), 4)
            _loss_d = round(float(loss_d.detach().cpu()), 4)
            _loss_r = float(loss_reconstruction.detach().cpu())

            if self._verbose:
                logger.info(
                    f"Epoch: {_epoch}, Loss G: {_loss_g: .4f}, "  # noqa: T001
                    f"Loss D: {_loss_d: .4f}, "
                    f"Loss R: {_loss_r: .4f}"
                )

            if self._epoch_callback is not None:
                self._epoch_callback(EpochInfo(_epoch, _loss_g, _loss_d, _loss_r))

    @random_state
    def sample(
        self,
        n: int,
        conditions: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Sample data similar to the training data.

        Providing conditions will increase the probability of producing the
        specified values in the key columns.

        Args:
            n: Number of rows to sample.
            conditions: If specified, dictionary mapping column names to column
                value to condition on. The returned DataFrame of ndarray is not
                guaranteed to have exactly the conditional values, but should
                produce them with higher probability than unconditional
                generation. NOTE: if you want to call this function directly,
                the column names are different, specifically, numeric columns
                must have '.value' appended to the name.

        Returns:
            numpy.ndarray or pandas.DataFrame in original representation
        """

        # TODO: The ANYWAY setup can condition on different values (or even
        # different columns) in a single batch, so eventually we may want to
        # bypass the groupby that leads to a single setting of conditions to be
        # used in this sample() function. Especially if force_conditioning=True
        # provides good quality, we could make conditional generation
        # substantially more efficient by converting the seed DataFrame directly
        # to batches of conditional vectors and do a single pass. No rejection
        # sampling, no groupby on (likely highly unique) numeric columns, etc.

        if conditions is not None:
            if self._conditional_vector_type == ConditionalVectorType.SINGLE_DISCRETE:
                # We could setup the discrete conditional vector if there's exactly
                # 1 seed column. But for now we keep previous behavior (due to
                # raising an error in actgan_wrapper.py) of no conditional
                # generation for SINGLE_DISCRETE, we rely entirely on rejection
                # sampling in the SDV code.
                fixed_cond_vec_torch = None
            else:
                cond_vec = self._transformer.convert_conditions(conditions)
                # Expand conditional vector to a full batch
                fixed_cond_vec_torch = torch.from_numpy(
                    np.repeat(cond_vec, repeats=self._batch_size, axis=0)
                ).to(self._device)
        else:
            if self._conditional_vector_type == ConditionalVectorType.SINGLE_DISCRETE:
                fixed_cond_vec_torch = None
            else:
                # For ANYWAY* conditional vectors, the unconditioned case always
                # uses the same cond vec of 0s.
                fixed_cond_vec_torch = torch.zeros(
                    (self._batch_size, self._cond_vec_dim),
                    dtype=torch.float32,
                    device=self._device,
                )

        # Switch generator to eval mode for inference
        self._generator.eval()
        steps = (n - 1) // self._batch_size + 1
        data = []
        for _ in range(steps):
            if fixed_cond_vec_torch is None:
                # In SINGLE_DISCRETE mode, so we generate a different cond vec
                # for every batch to match expected discrete distributions.
                cond_vec_numpy = self._condvec_sampler.sample_original_condvec(
                    self._batch_size
                )
                if cond_vec_numpy is not None:
                    cond_vec = torch.from_numpy(cond_vec_numpy).to(self._device)
                else:
                    cond_vec = None
            else:
                cond_vec = fixed_cond_vec_torch

            fakez = self._make_noise()

            _, fakeact = self._apply_generator(fakez, cond_vec)

            data.append(fakeact.detach().cpu().numpy())

        # Switch generator back to train mode now that inference is complete
        self._generator.train()
        data = np.concatenate(data, axis=0)
        data = data[:n]

        original_repr_data = self._transformer.inverse_transform(data)
        if self._force_conditioning and conditions is not None:
            # Bypass rejection sampling by directly setting the condition
            # columns to the requested values.
            for column_name, value in conditions.items():
                original_repr_data[column_name] = value

        return original_repr_data

    def set_device(self, device: str) -> None:
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
