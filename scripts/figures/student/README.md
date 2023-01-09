# Multivariate Student-t tasks

To visualise the correction term run:
```bash
$ python scripts/figures/student/correction.py
```

To generate tasks with varying degrees of freedom use
```bash
$ export STUDENTDIR=data/generated/student-t
$ export TASKDIR="$STUDENTDIR/tasks"
$ python scripts/figures/student/generate_tasks.py $TASKDIR
```

Now we can generate the full experimental design:
```bash
$ python scripts/experimental_design.py \
    --TASKS $TASKDIR \
    --ESTIMATORS scripts/figures/student/estimators.yaml \
    --RESULTS $STUDENTDIR/results
```

By a pipe to parallel (add ` | parallel` to the command) we can run it using GNU Parallel.

Once all the runs have finished, we can visualise them.

