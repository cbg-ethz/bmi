# BMI (Benchmarking Mutual Information)
Mutual information estimators and benchmark.

_The package is still in early development phase and the public API is very fluid. We hope it will stabilize in December 2022._

## Usage

To import the package (containing the estimators, samplers, and tasks) use:

    import bmi.api as bmi


## Contributing

Install the development requirements:

    $ pip install -r requirements.txt

Install the package in the editable mode, together with testing utilities:

    $ pip install -e ".[test]"

Install the pre-commit hooks:

    $ pre-commit install

At this stage it would be good to run unit tests:

    $ pytest


