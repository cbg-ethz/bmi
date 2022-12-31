# External estimators

In this directory we add wrappers around estimators
which are not part of the official Python package.

For example, they may use auxiliary dependencies or even be implemented in other languages.

## Available wrappers and estimators

### Julia and TransferEntropy.jl
In the file `mi_estimator.jl` we have a Julia wrapper around [TransferEntropy.jl](https://juliadynamics.github.io/TransferEntropy.jl/).

Note that you need to [install Julia](https://julialang.org/downloads/) first and then install the needed dependencies:
```
$ julia
julia> using Pkg
julia> Pkg.add(["ArgParse", "CSV", "DataFrames", "TransferEntropy"])
```

### R and the rmi package

In the file `rmi.R` we have an R wrapper around the [rmi package](https://cran.r-project.org/web/packages/rmi/index.html).

Note that you need to install R first and then install the necessary dependencies:
```
$ R
> install.packages("argparse", "dplyr", "rmi")
```

You can check if the dependencies have been installed properly by running
```
$ pytest tests/benchmark/test_wrapper --run-r
```
Note that without `--run-r` flag the tests will be skipped.

### MINE
As a dependency we use a [custom fork](https://github.com/pawel-czyz/mine-mist) of the [Latte project](https://github.com/boevalab/latte),
which contains MINE and MIST implementations (developed by Anej Svete).
The MINE implementation has been based on the original [Python + PyTorch implementation of MINE](https://github.com/gtegner/mine-pytorch).

The fork can be installed together with the package as an optional dependency:
```
$ pip install ".[mine]"
```

Alternatively, one can install it manually:
```
$ pip install git+https://github.com/pawel-czyz/mine-mist.git
```

You can run unit tests associated to it via:
```
$ pytest tests/estimators/external/test_mine.py --run-mine-pytorch
```
Note that without `--run-mine-pytorch` the tests will be skipped.
