# Multivariate Student-t tasks

To visualise the correction term run:
```bash
$ python scripts/figures/student/correction.py
```

To generate tasks with varying degrees of freedom use
```
$ export STUDENTDIR=data/generated/student-t
$ export TASKDIR="$STUDENTDIR/tasks"
$ python scripts/figures/student/generate_tasks.py $TASKDIR
```

