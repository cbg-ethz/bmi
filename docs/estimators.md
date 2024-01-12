# Estimators

The package supports a range of existing mutual information estimators. For the full list, see [below](#list-of-estimators).

## Example

The design of the estimators was motivated by [SciKit-Learn](https://scikit-learn.org/) API[@SciKit-Learn-API-2013].
All estimators are classes. Once a class is initialized, one can use the `estimate` method, which maps arrays containing data points (of shape `(n_points, n_dim)`) 
to mutual information estimates:

```python
import bmi

# Generate a sample with 1000 data points
task = bmi.benchmark.BENCHMARK_TASKS['1v1-normal-0.75']
X, Y = task.sample(1000, seed=42)
print(f"X shape: {X.shape}")  # Shape (1000, 1)
print(f"Y shape: {Y.shape}")  # Shape (1000, 1)

# Once an estimator is instantiated, it can be used to estimate mutual information
# by using the `estimate` method.
cca = bmi.estimators.CCAMutualInformationEstimator()
print(f"Estimate by CCA: {cca.estimate(X, Y):.2f}")

ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))
print(f"Estimate by KSG: {ksg.estimate(X, Y):.2f}")
```

Additionally, the estimators can be queried for their hyperparameters:
```python
print(cca.parameters())  # CCA does not have tunable hyperparameters
# _EmptyParams()

print(ksg.parameters())  # KSG has tunable hyperparameters
# KSGEnsembleParameters(neighborhoods=[5], standardize=True, metric_x='euclidean', metric_y='euclidean')
```

The returned objects are structured using [Pydantic](https://docs.pydantic.dev/).

## List of estimators

### Neural estimators

We support several standard neural estimators in [JAX](https://github.com/google/jax) basing on the [PyTorch implementations](https://github.com/ermongroup/smile-mi-estimator)[@Song-Ermon-2019]: 

  - Donsker-Varadhan estimator[@belghazi:18:mine] is implemented in [`DonskerVaradhanEstimator`](api/estimators.md#bmi.estimators.DonskerVaradhanEstimator).
  - MINE[@belghazi:18:mine] estimator, which is a Donsker-Varadhan estimator with correction debiasing gradient during the fitting phase, is implemented in [`MINEEstimator`](api/estimators.md#bmi.estimators.MINEEstimator).
  - InfoNCE[@oord:18:infonce], also known as Contrastive Predictive Coding, is implemented in [`InfoNCEEstimator`](api/estimators.md#bmi.estimators.InfoNCEEstimator).
  - NWJ estimator[@NWJ2007] is implemented as [`NWJEstimator`](api/estimators.md#bmi.estimators.NWJEstimator).

### Model-based estimators
  - Canonical correlation analysis[@Brillinger-2004,@kay-elliptic] is suitable when $P(X, Y)$ is multivariate normal and does not require hyperparameter tuning. It's implemented in [`CCAMutualInformationEstimator`](api/estimators.md#bmi.estimators.CCAMutualInformationEstimator).

### Histogram-based estimators
  - We implement a histogram-based estimator[@Cellucci-HistogramsMI] in [`HistogramEstimator`](api/estimators.md#bmi.estimators.HistogramEstimator). However, note that we do not support adaptive binning schemes.

## Kernel density estimators
  - We implement a simple kernel density estimator in [`KDEMutualInformationEstimator`](api/estimators.md#bmi.estimators.KDEMutualInformationEstimator).

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
4. Add the estimator to the [list of estimators](#list-of-estimators) and [ReadMe](index.md#list-of-estimators)


\bibliography