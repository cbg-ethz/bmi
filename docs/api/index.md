# API

## Tasks

[Tasks](tasks.md) represent named probability distributions which are used in the benchmark.

## Estimators

[Estimators](estimators.md) are the implemented mutual information estimators.

## Samplers

[Samplers](samplers.md) represent joint probability distributions with known mutual information from which one can sample. They are lower level than `Tasks` and can be used to define new tasks by transformations which preserve mutual information.

### Fine distributions
[Subpackage](fine-distributions.md) implementing distributions in which the ground-truth mutual information may not be known analytically, but can be efficiently approximated using Monte Carlo methods. 

## Interfaces
[Interfaces](interfaces.md) defines the main interfaces used in the package.
