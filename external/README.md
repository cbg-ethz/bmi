# External estimators

In this directory we add wrappers around estimators
which are not part of the official Python package.

For example, they may use auxiliary dependencies or even be implemented in other languages.

## Available wrappers and estimators

### Julia and TransferEntropy.jl
In the file `mi_estimator.jl` we have a Julia wrapper around [TransferEntropy.jl](https://juliadynamics.github.io/TransferEntropy.jl/).

Note that you need to [install Julia](https://julialang.org/downloads/) first and install the needed dependencies (instructions provided in the file).

### R and the rmi package

In the file `rmi.R` we have an R wrapper around the [rmi package](https://cran.r-project.org/web/packages/rmi/index.html).

Note that you need to install R first and install the necessary dependencies (instructions provided in the file).
