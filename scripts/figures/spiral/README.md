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

Then, run the estimators:
```
$ python scripts/figures/spiral/run_estimators.py $SPIRALDIR/tasks $SPIRALDIR/results 
```

Once the run has finished, you can generate the plot:
```
$ python scripts/figures/spiral/plot_performance.py $SPIRALDIR/tasks $SPIRALDIR/results $SPIRALDIR/figures/spiral_plot.pdf
```