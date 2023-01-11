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
$ python scripts/figures/spiral/generate_tasks.py $SPIRALDIR/tasks
```

Then, let's run the estimators. We'll use [GNU Parallel](https://www.gnu.org/software/parallel/) to do this.
Hence, we need to create an experimental design. Experimental design is a list of commands to be run
and explicitly contains all the estimators and the tasks used.

We will generate one using an auxiliary script `scripts/experimental_design.py` from
the task directory and a YAML file summarizing the estimators:
```bash
$ cat scripts/figures/spiral/estimators.yaml
```
Let's see a summary of the experimental design:
```bash
$ python scripts/experimental_design.py \
    --TASKS $SPIRALDIR/tasks \ 
    --ESTIMATORS scripts/figures/spiral/estimators.yaml \
    --summary
```
Now we now how many runs we expect. Let's generate the full experimental design:
```
$ python scripts/figures/spiral/experimental_design.py \
    --TASKS $SPIRALDIR/tasks \
    --RESULTS $SPIRALDIR/results \
    --ESTIMATORS scripts/figures/spiral/estimators.yaml
```
To actually run them, we need to pipe them to GNU Parallel:
```
$ python scripts/figures/spiral/experimental_design.py \
    --TASKS $SPIRALDIR/tasks \
    --RESULTS $SPIRALDIR/results \
    --ESTIMATORS scripts/figures/spiral/estimators.yaml | parallel
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
