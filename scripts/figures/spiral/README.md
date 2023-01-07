# Invariance to the spiral diffeomorphism

In this directory we store the scripts which can be used to reproduce the plot representing
whether the mutual information estimates are invariant to the spiral diffeomorphism.

  - `visualise_spiral.py`: plots a figure visualising spirals with different speed parameters.
  - `generate_spiral.py`: generates tasks with different speed parameters and saves them.
  - `run_estimators.py`: runs the estimators on the specified tasks and generates the results in the JSON format.
  - `plot_performance.py`: reads the generated results and plots them.


## Generating the plot
First, generate the directory
```
$ export SPIRALDIR=data/generated/spiral-figure
$ mkdir -p $SPIRALDIR/figures
```
Plot the spirals:
```
$ python scripts/figures/spiral/visualise_spiral.py $SPIRALDIR/figures/spiral_visualisation.pdf
```

Generate the benchmark tasks:
```
$ python scripts/figures/spiral/generate_spiral_tasks.py $SPIRALDIR/tasks
```

Then, let's run the estimators. We'll use [GNU Parallel](https://www.gnu.org/software/parallel/) to do this.
Hence, we need to create an experimental design. For this experiment, we created one in `scripts/figures/spiral/experimental_design.py`.
Let's take a look at it:
```
$ python scripts/figures/spiral/experimental_design.py $SPIRALDIR/tasks $SPIRALDIR/results --summary
```
Note the `--summary` flag, so only a summary is printed, rather than all the commands running the estimators!
To print out the commands run:
```
$ python scripts/figures/spiral/experimental_design.py $SPIRALDIR/tasks $SPIRALDIR/results
```
To actually run them, we need to pipe them to GNU Parallel:
```
$ python scripts/figures/spiral/experimental_design.py $SPIRALDIR/tasks $SPIRALDIR/results | parallel
```
Et voilà, we have all the estimators running in parallel, what is much faster than running them sequentially in Python!
You can observe how the new results appear every second by listing the `$SPIRALDIR/results` directory.
For example, I like running:
```
$ ls -1 $SPIRALDIR/results | wc -l
```
and compare it to expected number of runs from the summary.
Also, to see how many cores are used I recommend:
```
$ htop
```

Once all the runs have finished, you can generate the plot:
```
$ python scripts/figures/spiral/plot_performance.py $SPIRALDIR/tasks $SPIRALDIR/results $SPIRALDIR/figures/spiral_plot.pdf
```
