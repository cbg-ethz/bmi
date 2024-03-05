# Benchmarking a new estimator

Standard package utilities allow one to estimate mutual information on a given sample and can be convenient for the development of a new mutual information estimator.
However, once a new estimator is ready, how can it be benchmarked? Running it manually on many distributions can take a lot of effort.

Conveniently, [Snakemake](https://snakemake.readthedocs.io/en/stable/) is a workflow orchestrator which can generate benchmark tasks, save samples to the disk, and run many estimators.
Once Snakemake has been installed, we recommend running:

```bash
$ snakemake -c4 -s workflows/benchmark/demo/run.smk
```

In about a minute it should generate minibenchmark results in the `generated/benchmark/demo` directory. Note that the configuration file, `workflows/benchmark/demo/config.py`, explicitly defines the estimators and tasks used, as well as the number of samples.

Hence, it is possible to benchmark a *custom Python estimator* by simply importing it and including it in the configuration dictionary. Similarly, it is easy to change the number of samples or used tasks.
The `config.py` file is plain Python!

We defined several benchmark suites with the shared structure.

## Adding a new Python estimator

Every added estimator has to implement the [`IMutualInformationPointEstimator`](https://github.com/cbg-ethz/bmi/blob/main/src/bmi/interface.py) interface.

Let's take a look at the simplest possible estimator (which generally shouldn't be used in practice), which estimates mutual information as $\hat I(X; Y) = -0.5 \log\left( 1-\mathrm{Corr}(X_1, Y_1)^2 \right)$.

```{python}
import numpy as np

from bmi.interface import BaseModel, IMutualInformationPointEstimator

class EmptyParams(BaseModel):
    """This estimator has no hyperparameters."""
    pass

class TooSimpleEstimator(IMutualInformationPointEstimator):
    def __init__(self) -> None:
        """All estimator hyperparameters should be set at this stage.
        This estimator does not have any hyperparameters, though.
        """
        pass

    def estimate(self, x, y) -> float:
        """Estimates mutual information.

        Attrs:
            x: samples from the X variable, shape (n_points, dim_x)
            y: samples from the Y variable, shape (n_points, dim_y)
        """
        x, y = np.asarray(x), np.asarray(y)
        x1 = x[:, 0]
        y1 = y[:, 0]

        rho = np.corrcoef(x1, y1)[0, 1]
        return -0.5 * np.log(1 - rho**2)

    def parameters(self) -> BaseModel:
        """Returns the hyperparameters of the estimator."""
        return EmptyParams()
```

If your estimator is a function, you can also wrap it into a class using [`FunctionalEstimator`](https://github.com/cbg-ethz/bmi/blob/main/src/bmi/estimators/function_wrapper.py) wrapper.

Once such a class is available, it can be simply included into the configuration dictionary.

## Adding a non-Python estimator

How can one use the benchmark to evaluate an estimator implemented in another programming language?
The benchmark employs Snakemake workflows in which each estimator has to implement the `IMutualInformationPointEstimator` interface.

However, it's possible to create a thin wrapper, which whenever called will execute the following steps:

- Save the samples to a temporary location.
- Run an external script (e.g., in Julia, R, C++ or any other language) which prints mutual information estimate to the standard output.
- Convert the standard output to `float` and return it as the estimate.

The external script can be implemented in any language. Generally it will take three required arguments:

1. The path to the CSV file with samples.
2. Number of columns representing the $X$ variable, `dim_x`.
3. Number of columns representing the $Y$ variable, `dim_y`.

Additionally, it can take any additional arguments controlling the hyperparameters.
Example scripts running several existing estimators implemented in Julia and R are [here](https://github.com/cbg-ethz/bmi/tree/main/external).

Once a script is ready and can be run manually on CSV samples, it can be wrapped into a Python class implementing the `IMutualInformationPointEstimator` interface.
Routines such as saving the samples to a CSV file or checking the standard output are standard, so we implemented a convenient class, [`ExternalEstimator`](https://github.com/cbg-ethz/bmi/blob/main/src/bmi/estimators/external/external_estimator.py).

For example, let's take a look how the wrapper around the KSG estimator in R can be implemented.

```python

from pathlib import Path

from bmi.estimators.external.external_estimator import ExternalEstimator
from bmi.interface import BaseModel, Pathlike

R_PATH = "path_to_the_R_script"


class KSGParams(BaseModel):
    """The hyperparameters of a given estimator.

    Attrs:
        neighbors (int): Number of neighbors to be used in the KSG algorithm
    """
    neighbors: int


class RKSGEstimator(ExternalEstimator):
    """The KSG estimators implemented in the `rmi` package in R."""

    def __init__(self, neighbors: int = 10) -> None:
        """
        Args:
            neighbors: number of neighbors (k) to be used
        """
        self._params = KSGParams(
            variant=variant,
            neighbors=neighbors,
        )

    def parameters(self) -> KSGParams:
        """Returns the hyperparameters of the estimator."""
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int) -> list[str]:
        sample_path_abs = str(Path(path).absolute())
        return [
            "Rscript",
            estimator_r_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--method",
            f"KSG1",
            "--neighbors",
            str(self._params.neighbors),
        ]


estimator = RKSGEstimator(neighbors=5)
# This is an ordinary estimator now and can be run with `estimator.estimate(X, Y)`
```

More Python wrappers around external scripts are implemented [here](https://github.com/cbg-ethz/bmi/tree/main/src/bmi/estimators/external).

## FAQ

### How to add an estimator implemented in another programming language?

It's possible to quickly implement a thin Python wrapper around an estimator implemented in another language.
See [this section](#adding-a-non-python-estimator) for details.

