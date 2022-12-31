"""The MINE estimator.

Note:
    This module uses optional dependencies,
    so it should NEVER be imported by any of the core modules and APIs.
"""
import enum
from typing import Literal, Sequence

from numpy.typing import ArrayLike

try:
    # pytype: disable=import-error
    import torch
    import torch.nn as nn
    from latte.models.mine import mine
    from latte.modules.data import datamodules as dm
    from pytorch_lightning import Trainer

    # pytype: enable=import-error
except ImportError as e:
    raise ImportError(
        f"It seems that you are missing a dependency."
        f'Try calling `pip install ".[mine]"`. Raised error {e}.'
    )

from bmi.estimators.base import EstimatorNotFittedException
from bmi.interface import IMutualInformationPointEstimator
from bmi.utils import ProductSpace


def _parse_layers(dimensions: Sequence[int]) -> list[tuple[int, int]]:
    if len(dimensions) < 2:
        raise ValueError("At least two dimensions must be specified (input and output).")

    layers = []
    for i in range(len(dimensions) - 1):
        input_dim = dimensions[i]
        output_dim = dimensions[i + 1]
        layers.append((input_dim, output_dim))
    return layers


def _construct_statistics_network(
    dim_x: int,
    dim_y: int,
    hidden_layers: Sequence[int] = (5,),
    bias: bool = True,
    activation: nn.Module = nn.ReLU(),
) -> mine.StatisticsNetwork:
    """Constructs the statistics network.

    Args:
        dim_x: dimension of the X variable
        dim_y: dimensions of the Y variable
        hidden_layers: dimensions of the hidden linear layers of the statistics neural network
        bias: whether to use bias in the linear layers
        activation: non-linear activation function to be used
    """
    layers_spec = _parse_layers([dim_x + dim_y] + list(hidden_layers))
    layers = []
    for input_dim, output_dim in layers_spec:
        layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        layers.append(activation)

    return mine.StatisticsNetwork(
        S=nn.Sequential(*layers),
        out_dim=hidden_layers[-1],
    )


def _unpack_result(wrapped) -> float:
    """The numerical MI estimate we retrieve is `wrapped` as:
        [{'some_name': 0.002}]
    the point is now to find this 0.002.
    """
    vals = list(wrapped[0].values())
    if len(vals) != 1:
        raise ValueError(f"Item {wrapped} is not formatted appropriately.")
    return vals[0]


class MINEObjectiveType(enum.Enum):
    MINE = "mine"
    F_DIV = "f-div"
    MINE_BIASED = "mine-biased"


def _parse_objective_type(objective: MINEObjectiveType) -> mine.MINEObjectiveType:
    if objective == objective.MINE:
        return mine.MINEObjectiveType.MINE
    elif objective.MINE_BIASED:
        return mine.MINEObjectiveType.MINE_BIASED
    elif objective == objective.F_DIV:
        return mine.MINEObjectiveType.F_DIV
    else:
        raise ValueError("Objective type not recognized.")


class MutualInformationNeuralEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        # Data parameters
        standardize: bool = True,
        proportion_train: float = 1 / 3,
        proportion_valid: float = 1 / 3,
        # MINE parameters
        objective: MINEObjectiveType = MINEObjectiveType.MINE,
        alpha: float = 1e-2,
        # Statistics NN parameters
        hidden_layers: Sequence[int] = (10, 5),
        bias: bool = True,
        # Training parameters
        learning_rate: float = 1e-3,
        max_epochs: int = 300,
        batch_size: int = 32,
        device: Literal["cpu", "gpu"] = "cpu",
        seed: int = 714,
    ) -> None:
        """

        Args:
            standardize: whether to standardize the data (recommended)
            proportion_train: specifies the fraction of samples to go into the training data set
            proportion_valid: specifies the fraction of samples to go into the validation data set
            hidden_layers: specifices the layers of the statistics NN
            bias: specifies the bias in the statistics NN
            seed: random seed for PyTorch
        """
        # Whether the model has been fitted to the data
        # and can be queried about the training
        self._fitted: bool = False
        self._trainer = None
        self._data_module = None
        self._results = None

        # Data processing parameters
        self._standardize = standardize
        self._proportion_train = proportion_train
        self._proportion_valid = proportion_valid
        self._proportion_test = 1 - (proportion_train + proportion_valid)

        if min(self._proportion_train, self._proportion_valid, self._proportion_test) < 0:
            raise ValueError("The proportion of each data set must be positive.")

        # MINE parameters
        self._alpha = alpha
        self._objective = objective

        # Statistics network parameters
        self._hidden_layers = hidden_layers
        self._bias = bias

        # Training parameters
        if max_epochs < 1:
            raise ValueError("Maximum number of methods must be at least 1.")
        self._max_epochs = max_epochs
        self._device = device
        self._learning_rate = learning_rate
        if learning_rate < 0:
            raise ValueError("Learning rate must be positive.")
        self._batch_size = batch_size
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        self._seed = seed

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        space = ProductSpace(x=x, y=y, standardize=self._standardize)

        torch.manual_seed(self._seed)

        data_module = dm.GenericDataModule(
            X=torch.from_numpy(space.x).float(),
            Y=torch.from_numpy(space.y).float(),
            p_train=self._proportion_train,
            p_val=self._proportion_valid,
            batch_size=self._batch_size,
        )

        network = _construct_statistics_network(
            dim_x=space.dim_x,
            dim_y=space.dim_y,
            hidden_layers=self._hidden_layers,
            bias=self._bias,
        )
        model = mine.StandaloneMINE(
            T=network,
            kind=_parse_objective_type(self._objective),
            alpha=self._alpha,
        )
        trainer = Trainer(
            max_epochs=self._max_epochs,
            enable_model_summary=False,
            enable_progress_bar=False,
            accelerator=self._device,
        )
        trainer.fit(model, datamodule=data_module)

        results = {
            "test": _unpack_result(
                trainer.test(ckpt_path="best", dataloaders=data_module, verbose=False)
            ),
            "valid": _unpack_result(
                trainer.validate(ckpt_path="best", dataloaders=data_module, verbose=False)
            ),
            "train": _unpack_result(
                trainer.validate(
                    ckpt_path="best", dataloaders=data_module.train_dataloader(), verbose=False
                )
            ),
        }

        self._trainer = trainer
        self._data_module = data_module
        self._results = results
        self._fitted = True

    def get_trainer(self) -> Trainer:
        if not self._fitted or self._trainer is None:
            raise EstimatorNotFittedException
        return self._trainer

    def get_data_module(self):
        return self._data_module

    def mi_train(self) -> float:
        if not self._fitted or self._results is None:
            raise EstimatorNotFittedException
        return self._results["train"]

    def mi_valid(self) -> float:
        if not self._fitted or self._results is None:
            raise EstimatorNotFittedException
        return self._results["valid"]

    def mi_test(self) -> float:
        if not self._fitted or self._results is None:
            raise EstimatorNotFittedException
        return self._results["test"]

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        self.fit(x, y)
        return self.mi_test()
