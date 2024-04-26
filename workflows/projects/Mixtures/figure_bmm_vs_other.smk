# Figure comparing BMMs and other estimators on selected problems.
# Note: to run this workflow, you need to have the results.csv files from:
#     - the benchmark (version 2) in `generated/benchmark/v2/results.csv`
#     - the BMM minibenchmark in `generated/projects/Mixtures/gmm_benchmark/results.csv`  

rule all:
    input: "generated/projects/Mixtures/figure_bmm_vs_other.pdf"

rule generate_figure:
    output: "generated/projects/Mixtures/figure_bmm_vs_other.pdf"
    input:
        v2 = "generated/benchmark/v2/results.csv",
        bmm = "generated/projects/Mixtures/gmm_benchmark/results.csv"
    run:
        raise NotImplementedError
