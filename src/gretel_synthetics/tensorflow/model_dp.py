from typing import Tuple, TYPE_CHECKING

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

if TYPE_CHECKING:
    from gretel_synthetics.config import TensorFlowConfig
else:
    TensorFlowConfig = None


def compute_epsilon(steps: int, store: TensorFlowConfig, epoch_number: int = None) -> Tuple[float, float]:
    """
    Calculate epsilon and delta values for differential privacy

    Returns:
        Tuple of eps, opt_order
    """
    # Note: inverse of number of training samples recommended for minimum
    # delta in differential privacy
    if epoch_number is None:
        epoch_number = store.epochs - 1
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=steps,
        batch_size=store.batch_size,
        noise_multiplier=store.dp_noise_multiplier,
        epochs=epoch_number,
        delta=1.0 / float(steps),
    )
