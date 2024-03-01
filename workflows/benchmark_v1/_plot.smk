import _plot_utils as utils

rule plot_benchmark:
    input: 'results.csv'
    output: 'figures/benchmark.pdf'
    run:
        results = utils.read_results(str(input))
        fig, ax = utils.subplots_benchmark(results)
        utils.plot_benchmark(ax, results)
        fig.savefig(str(output))
