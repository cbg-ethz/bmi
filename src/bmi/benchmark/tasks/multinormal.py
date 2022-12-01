import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import generate_task

sampler = samplers.SplitMultinormal(covariance=np.eye(3), dim_x=1, dim_y=2)
seeds = np.arange(10)

task_mn_1_2_eye_100 = generate_task(
    sampler=sampler, n_samples=100, seeds=seeds, task_id="mn-1-2-eye-100"
)

task_mn_1_2_eye_1000 = generate_task(
    sampler=sampler, n_samples=1000, seeds=seeds, task_id="mn-1-2-eye-1000"
)
