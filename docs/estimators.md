# Estimators

The package supports a range of existing mutual information estimators. For the full list, see [below](#list-of-estimators).

## General usage instructions

## List of estimators

### Neural estimators
  - MINE[@belghazi:18:mine] estimator is implemented as [`MINEEstimator`](api/estimators.md#bmi.estimators.MINEEstimator).
  - InfoNCE[@oord:18:infonce], also known as Contrastive Predictive Coding is implemented as [`InfoNCEEstimator`](api/estimators.md#bmi.estimators.InfoNCEEstimator).

### Model-based estimators
  - Canonical correlation analysis[@Brillinger-2004,@kay-elliptic]

### Histogram-based estimators

### Neighborhood-based estimators
  - An ensemble of Kraskov-Stögbauer-Grassberger estimators[@kraskov:04:ksg] is implemented as [`KSGEnsembleFirstEstimator`](api/estimators.md#bmi.estimators.KSGEnsembleFirstEstimator).

## FAQ

### Do these estimators work for discrete variables?
When both variables $X$ and $Y$ are discrete, we recommend the [`dit` package](https://github.com/dit/dit). When one variable is discrete and the other is continuous, one can approximate mutual information by adding small noise to the discrete variable. 

!!! todo

    Add a Python example showing how to add the noise.

### Where is the API showing how to use the estimators?

The API is [here](api/estimators.md).

### How can I add a new estimator?
Thank you for considering contributing to this project! Please, consult [contributing guidelines](contributing.md) and reach out to us on [GitHub](https://github.com/cbg-ethz/bmi/issues), so we can discuss the best way of adding the estimator to the package.
Generally, the following steps are required:
1. Implement the interface [`IMutualInformationPointEstimator`](api/interfaces.md#bmi.interface.IMutualInformationPointEstimator) in a new file inside `src/bmi/estimators` directory. The unit tests should be added in `tests/estimators` directory.
2. Export the new estimator to the public API by adding an entry in `src/bmi/estimators/__init__.py`.
3. Export the docstring of new estimator to `docs/api/estimators.md`.
4. Add the estimator to the [list of estimators](#list-of-estimators).

\bibliography