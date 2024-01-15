[![arXiv](https://img.shields.io/badge/arXiv-2306.11078-b31b1b.svg)](https://arxiv.org/abs/2306.11078)
[![Venue](https://img.shields.io/badge/venue-NeurIPS_2023-darkblue)](https://neurips.cc/virtual/2023/poster/72978)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PyPI Latest Release](https://img.shields.io/pypi/v/benchmark-mi.svg)](https://pypi.org/project/benchmark-mi/)
[![build](https://github.com/cbg-ethz/bmi/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/cbg-ethz/bmi/actions/workflows/build.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Benchmarking Mutual Information

BMI is the package for estimation of mutual information between continuous random variables and testing new estimators.

- **Documentation**: [https://cbg-ethz.github.io/bmi/](https://cbg-ethz.github.io/bmi/)
- **Source code**: [https://github.com/cbg-ethz/bmi](https://github.com/cbg-ethz/bmi)
- **Bug reports**: [https://github.com/cbg-ethz/bmi/issues](https://github.com/cbg-ethz/bmi/issues)
- **PyPI package**: [https://pypi.org/project/benchmark-mi](https://pypi.org/project/benchmark-mi)

## Getting started
While we recommend taking a look at the [documentation](https://cbg-ethz.github.io/bmi/) to learn about full package capabilities, below we present the main capabilities of the Python package.
(Note that BMI can also be used to test non-Python mutual information estimators.)

You can install the package using:

```bash
$ pip install benchmark-mi
```

Alternatively, you can use the development version from source using:

```bash
$ pip install "bmi @ https://github.com/cbg-ethz/bmi"
```

Note: BMI uses [JAX](https://github.com/google/jax) and by default installs the CPU version of it.
If you have a device supporting CUDA, you can [install the CUDA version of JAX](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).

Now let's take one of the predefined distributions included in the benchmark (named "tasks") and sample 1,000 data points.
Then, we will run two estimators on this task.

```python
import bmi

task = bmi.benchmark.BENCHMARK_TASKS['1v1-normal-0.75']
print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
print(f"Ground truth mutual information: {task.mutual_information:.2f}")

X, Y = task.sample(1000, seed=42)

cca = bmi.estimators.CCAMutualInformationEstimator()
print(f"Estimate by CCA: {cca.estimate(X, Y):.2f}")

ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))
print(f"Estimate by KSG: {ksg.estimate(X, Y):.2f}")
```

## Citing

If you find this code useful in your research, consider citing [our manuscript](https://arxiv.org/abs/2306.11078):

```
@inproceedings{beyond-normal-2023,
  title={Beyond Normal: On the Evaluation of Mutual Information Estimators},
  author={Paweł Czyż and Frederic Grabowski and Julia E. Vogt and Niko Beerenwinkel and Alexander Marx},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## List of estimators

- The neighborhood-based KSG estimator proposed in [Estimating Mutual Information](https://arxiv.org/abs/cond-mat/0305641) by Kraskov et al. (2003).
- Donsker-Varadhan and MINE estimators proposed in [MINE: Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) by Belghazi et al. (2018).
- InfoNCE estimator proposed in [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) by Oord et al. (2018).
- NWJ estimator proposed in [Estimating divergence functionals and the likelihood ratio by convex risk minimization](https://arxiv.org/abs/0809.0853) by Nguyen et al. (2008).
- Estimator based on canonical correlation analysis described in [Feature discovery under contextual supervision using mutual information](https://ieeexplore.ieee.org/document/227286) by Kay (1992) and in [Some data analyses using mutual information](https://www.jstor.org/stable/43601047) by Brillinger (2004).
