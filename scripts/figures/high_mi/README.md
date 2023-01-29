# Changing MI and number of samples

To generate tasks with changing MI:
```bash
$ python scripts/figures/high_mi/generate_tasks.py data/generated/changing-mi-20230118/tasks \
  --CHANGE MI
Exception in the sparse Student task with mi =0, n_samples =5000, df=7, dim=3 due to The desired accuracy has not been reached..
Exception in the sparse Student task with mi =0.5, n_samples =5000, df=7, dim=3 due to The desired accuracy has not been reached..
```

To generate tasks with changing the number of samples:
```bash
$ python scripts/figures/high_mi/generate_tasks.py data/generated/changing-samples-20230118/tasks \
  --CHANGE SAMPLES
```

Now generate the experimental designs:

```bash
$ python scripts/experimental_design.py \
  --TASKS data/generated/changing-mi-20230118/tasks \
  --ESTIMATORS scripts/figures/high_mi/estimators.yaml \ 
  --RESULTS data/generated/changing-mi-20230118/results > changing_mi_design.txt
```

```bash
$ python scripts/experimental_design.py \
  --TASKS data/generated/changing-samples-20230118/tasks \
  --ESTIMATORS scripts/figures/high_mi/estimators.yaml \ 
  --RESULTS data/generated/changing-samples-20230118/results > changing_samples_design.txt
```

... and we can run them overnight:

```bash
$ cat changing_mi_design.txt changing_samples_design.txt | parallel -n 4
```
