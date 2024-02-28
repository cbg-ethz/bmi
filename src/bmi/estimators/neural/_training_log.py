"""Utility class for keeping information about training and displaying tqdm."""

from typing import Union

import jax
import jax.numpy as jnp
import tqdm


class TrainingLog:
    def __init__(
        self,
        max_n_steps: int,
        early_stopping: bool,
        train_smooth_factor: float = 0.1,
        verbose: bool = True,
        enable_tqdm: bool = True,
        train_history_in_additional_information: bool = False,
        test_history_in_additional_information: bool = True,
    ) -> None:
        """
        Args:
            max_n_steps: maximum number of training steps allowed
            early_stopping: whether early stopping is turned on
            train_smooth_factor: fraction of the training history length used
               as the smoothing window for convergence checks. E.g. when
               `train_smooth_factor=0.1` and the current training history
               has length 1000, averages over 100 steps will be used. Max 0.5.
            verbose: whether to print information during the training
            enable_tqdm: whether to use tqdm's progress bar during training
            history_in_additional_information: whether the generated additional
              information should contain training history (evaluated loss on
              training and test populations). We recommend keeping this flag
              turned on.
        """
        assert train_smooth_factor <= 0.5, "train_smooth_factor can be at most 0.5"

        self.max_n_steps = max_n_steps
        self.early_stopping = early_stopping
        self.train_smooth_factor = train_smooth_factor
        self.verbose = verbose

        self._train_history_in_additional_information = train_history_in_additional_information
        self._test_history_in_additional_information = test_history_in_additional_information

        self._mi_train_history: list[tuple[int, float]] = []
        self._mi_test_history: list[tuple[int, float]] = []
        self._mi_test_best = None
        self._logs_since_mi_test_best = 0
        self._tqdm = None
        self._additional_information = {}

        if verbose and enable_tqdm:
            self._tqdm_init()

    def log_train_mi(self, n_step: int, mi: Union[float, jax.Array]) -> None:
        """
        Args:
            mi: float or JAX's float-like, e.g., Array(0.5)
        """
        self._mi_train_history.append((n_step, float(mi)))
        self._tqdm_update()

    def log_test_mi(self, n_step: int, mi: Union[float, jax.Array]) -> None:
        """
        Args:
            mi: float or JAX's float-like, e.g., Array(0.5)
        """
        if self._mi_test_best is None or self._mi_test_best < mi:
            self._mi_test_best = mi
            self._logs_since_mi_test_best = 0
        else:
            self._logs_since_mi_test_best += 1

        self._mi_test_history.append((n_step, float(mi)))

        if self.verbose and self._tqdm is None:
            print(f"MI test: {mi:.2f} (step={n_step})")

        self._tqdm_refresh()

    @property
    def final_mi(self) -> float:
        if self._mi_test_best is None:
            return float("nan")

        return self._mi_test_best

    @property
    def additional_information(self) -> dict:
        if self._mi_train_history:
            n_steps, _ = self._mi_train_history[-1]
        else:
            n_steps = 0

        # Additional information we can return
        info = self._additional_information | {
            "n_training_steps": n_steps,
        }

        if self._train_history_in_additional_information:
            info |= {"training_history": self._mi_train_history}
        if self._test_history_in_additional_information:
            info |= {"test_history": self._mi_test_history}

        return info

    def early_stop(self) -> bool:
        return self.early_stopping and self._logs_since_mi_test_best > 1

    def finish(self):
        self._tqdm_close()
        self.detect_warnings()

    def detect_warnings(self):  # noqa: C901
        # early stopping
        if self.early_stopping and not self.early_stop():
            self._additional_information["early_stopping_not_triggered"] = True
            if self.verbose:
                print("WARNING: Early stopping enabled but max_n_steps reached.")

        # get train MI history
        train_mi = jnp.array([mi for _step, mi in self._mi_train_history])
        w = int(self.train_smooth_factor * len(train_mi))

        # check if training long enough to compute diagnostics
        if w < 1:
            self._additional_information["training_too_short_for_diagnostics"] = True
            if self.verbose:
                print("WARNING: Training too short to compute diagnostics.")
            return

        # compute smoothed mi
        cs = jnp.cumsum(jnp.concatenate([jnp.zeros(1), train_mi]))
        train_mi_smooth = (cs[w:] - cs[:-w]) / w

        # n + 1 - w >= w + 1 since w <= int(0.5 * n)
        assert len(train_mi_smooth) >= w + 1

        train_mi_smooth_max = float(train_mi_smooth.max())
        train_mi_smooth_fin = float(train_mi_smooth[-1])
        if train_mi_smooth_max > 1.05 * train_mi_smooth_fin:
            self._additional_information["max_training_mi_decreased"] = True
            if self.verbose:
                print(
                    f"WARNING: Smoothed training MI fell compared to highest value: "
                    f"max={train_mi_smooth_max:.3f} vs "
                    f"final={train_mi_smooth_fin:.3f}"
                )

        train_mi_smooth_fin = float(train_mi_smooth[-1])
        train_mi_smooth_prv = float(train_mi_smooth[-w])
        if train_mi_smooth_fin > 1.05 * train_mi_smooth_prv:
            self._additional_information["training_mi_still_increasing"] = True
            if self.verbose:
                print(
                    f"WARNING: Smoothed raining MI was still "
                    f"increasing when training stopped: "
                    f"final={train_mi_smooth_fin:.3f} vs "
                    f"{w} step(s) ago={train_mi_smooth_prv:.3f}"
                )

    def _tqdm_init(self):
        self._tqdm = tqdm.tqdm(
            total=self.max_n_steps,
            unit="step",
            ncols=120,
        )

    def _tqdm_update_prefix(self):
        if self._tqdm is None:
            return

        if self._mi_train_history:
            train_str = f"{self._mi_train_history[-1][-1]:.2f}"
        else:
            train_str = "???"

        if self._mi_test_history:
            test_str = f"{self._mi_test_history[-1][-1]:.2f}"
        else:
            test_str = "???"

        self._tqdm.set_postfix(train=train_str, test=test_str)

    def _tqdm_update(self):
        if self._tqdm is not None:
            self._tqdm_update_prefix()
            self._tqdm.update()

    def _tqdm_refresh(self):
        if self._tqdm is not None:
            self._tqdm_update_prefix()
            self._tqdm.refresh()

    def _tqdm_close(self):
        if self._tqdm is None:
            return

        self._tqdm.close()
        self._tqdm = None
