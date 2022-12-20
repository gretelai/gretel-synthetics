import logging

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from gretel_synthetics.actgan.base import BaseSynthesizer, random_state
from gretel_synthetics.actgan.data_sampler import DataSampler
from gretel_synthetics.actgan.data_transformer import DataTransformer
from gretel_synthetics.actgan.structures import ActivationFn, EpochInfo
from gretel_synthetics.typing import DFLike
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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        assert input_.size()[0] % self.pac == 0
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

    def __init__(self, embedding_dim, generator_dim, data_dim):
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


class ACTGANSynthesizer(BaseSynthesizer):
    """Anyway Conditional Table GAN Synthesizer.

    This is the core class of the ACTGAN interface.

    Args:
        embedding_dim:
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim:
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim:
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr:
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay:
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr:
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay:
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size:
            Number of data samples to process in each step.
        discriminator_steps:
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        binary_encoder_cutoff:
            For any given column, the number of unique values that should exist before
            switching over to binary encoding instead of OHE. This will help reduce
            memory consumption for datasets with a lot of unique values.
        binary_encoder_nan_handler:
            Binary encoding currently may produce errant NaN values during reverse transformation. By default
            these NaN's will be left in place, however if this value is set to "mode" then those NaN's will
            be replaced by a random value that is a known mode for a given column.
        log_frequency:
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose:
            Whether to have log progress results. Defaults to ``False``.
        epochs:
            Number of training epochs. Defaults to 300.
        epoch_callback:
            If set to a callable, call the function with `EpochInfo` as the arg
        pac:
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda:
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Sequence[int] = (256, 256),
        discriminator_dim: Sequence[int] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        binary_encoder_cutoff: int = 500,
        binary_encoder_nan_handler: Optional[str] = None,
        log_frequency: bool = True,
        verbose: bool = False,
        epochs: int = 300,
        epoch_callback: Optional[Callable] = None,
        pac: int = 10,
        cuda: bool = True,
    ):

        if batch_size % 2 != 0:
            raise ValueError("`batch_size` must be divisible by 2")

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
        self._verbose = verbose
        self._epochs = epochs
        self._epoch_callback = epoch_callback
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        if self._epoch_callback is not None and not callable(self._epoch_callback):
            raise ValueError("`epoch_callback` must be a callable or `None`")

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
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
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(
                    logits, tau=tau, hard=hard, eps=eps, dim=dim
                )
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == ActivationFn.TANH:
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == ActivationFn.SIGMOID:
                    ed = st + span_info.dim
                    data_t.append(torch.sigmoid(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == ActivationFn.SOFTMAX:
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(
                        f"Unexpected activation function {span_info.activation_fn}."
                    )

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if (
                    len(column_info) != 1
                    or span_info.activation_fn != ActivationFn.SOFTMAX
                ):
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

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
    ) -> np.ndarray:
        if discrete_columns is None:
            discrete_columns = ()

        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer(
            binary_encoder_cutoff=self._binary_encoder_cutoff,
            binary_encoder_nan_handler=self._binary_encoder_nan_handler,
            verbose=self._verbose,
        )
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        return train_data

    def _actual_fit(self, train_data: DFLike) -> None:
        """Fit the ACTGAN Synthesizer models to the training data.

        Args:
            train_data: Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        """

        epochs = self._epochs

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
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

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for _ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            _epoch = i + 1
            _loss_g = round(float(loss_g.detach().cpu()), 4)
            _loss_d = round(float(loss_d.detach().cpu()), 4)

            if self._verbose:
                logger.info(
                    f"Epoch: {_epoch}, Loss G: {_loss_g: .4f}, "  # noqa: T001
                    f"Loss D: {_loss_d: .4f}",
                )

            if self._epoch_callback is not None:
                self._epoch_callback(EpochInfo(_epoch, _loss_g, _loss_d))

    @random_state
    def sample(
        self,
        n: int,
        condition_column: Optional[str] = None,
        condition_value: Optional[str] = None,
    ) -> pd.DataFrame:
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n: Number of rows to sample.
            condition_column: Name of a discrete column.
            condition_value: Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = (
                self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, self._batch_size
                )
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        transformed_data = self._transformer.inverse_transform(data)
        return transformed_data

    def set_device(self, device: str) -> None:
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
