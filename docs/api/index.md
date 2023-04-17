# API

- [Tasks](tasks.md) represent named probability distributions which are used in the benchmark.
- [Estimators](estimators.md) are the implemented mutual information estimators.
- [Samplers](samplers.md) represent joint probability distributions with known mutual information from which one can sample. They are lower level than `Tasks` and can be used to define new tasks by transformations which preserve mutual information.
