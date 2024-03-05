# Demo benchmark

This is a microbenchmark used for demonstratory purposes. It consists of two files:

- `config.py`: defines estimators and tasks used. This file can be conveniently adjusted to include more estimators or different tasks and sample sizes.
- `run.smk`: Snakemake rules configured via `config.py`. This file is not necessary to modify.

## Running instructions

First, install [Snakemake](https://snakemake.readthedocs.io/en/stable/).
Then, you can run the demo benchmark suite (from the root project directory) using:

```bash
$ snakemake -c4 -s workflows/benchmark/demo/run.smk
```

This workflow should finish under a minute and generate benchmark results in `generated/benchmark/demo` directory.

