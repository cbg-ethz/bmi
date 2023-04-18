[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Benchmarking Mutual Information

BMI is the package for estimation of mutual information between continuous random variables and testing new estimators.

- **Documentation**: [https://cbg-ethz.github.io/bmi/](https://cbg-ethz.github.io/bmi/)
- **Source code**: [https://github.com/cbg-ethz/bmi](https://github.com/cbg-ethz/bmi)
- **Bug reports**: [https://github.com/cbg-ethz/bmi/issues](https://github.com/cbg-ethz/bmi/issues)


## Getting started
While we recommend taking a look at the [documentation](https://cbg-ethz.github.io/bmi/) to learn about full package capabilities, below we present the main capabilities of the Python package.
(Note that BMI can also be used to test non-Python mutual information estimators.)

You can install the package using:

**TODO:** Add installation instructions after we push to PyPI.
```bash
$ pip install "bmi @ https://github.com/cbg-ethz/bmi"
```

Note: BMI uses [JAX](https://github.com/google/jax) and by default installs the CPU version of it.
If you have a device supporting CUDA, you can [install the CUDA version of JAX](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).

Now let's take one of the predefined distributions included in the benchmark (named "tasks") and sample 1,000 data points.
Then, we will run two estimators on this task.

```python
import bmi

task = bmi.benchmark.BENCHMARK_TASKS['multinormal-dense-2-5-0.5']
print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
print(f"Ground truth mutual information: {task.mutual_information():.2f}")

X, Y = task.sample(1000, seed=42)

cca = bmi.estimators.CCAMutualInformationEstimator()
print(f"Estimate by CCA: {cca.estimate(X, Y):.2f}")

ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))
print(f"Estimate by KSG: {ksg.estimate(X, Y):.2f}")
```

## Citing

**TODO:** Add link to arXiv preprint and generate a BibTeX entry.
