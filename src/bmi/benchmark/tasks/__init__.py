from bmi.benchmark.tasks.additive_noise import task_additive_noise
from bmi.benchmark.tasks.bimodal_gaussians import task_bimodal_gaussians
from bmi.benchmark.tasks.bivariate_normal import task_bivariate_normal
from bmi.benchmark.tasks.embeddings import transform_swissroll_task
from bmi.benchmark.tasks.half_cube import transform_half_cube_task

# isort: off
from bmi.benchmark.tasks.multinormal import (
    task_multinormal_dense,
    task_multinormal_sparse,
    task_multinormal_2pair,
    task_multinormal_lvm,
)

from bmi.benchmark.tasks.mixtures import (
    task_x,
    task_ai,
    task_waves,
    task_galaxy,
    task_concentric_multinormal,
    task_multinormal_sparse_w_inliers,
)

# isort: on
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task
from bmi.benchmark.tasks.rotate import transform_rotate_task

# isort: off
from bmi.benchmark.tasks.scaling import (
    transform_gaussianise,
    transform_rescale,
    transform_uniformise,
)

# isort: on
from bmi.benchmark.tasks.spiral import transform_spiral_task

# isort: off
from bmi.benchmark.tasks.student import (
    task_student_dense,
    task_student_sparse,
    task_student_2pair,
)

# isort: on
from bmi.benchmark.tasks.wiggly import transform_wiggly_task

__all__ = [
    "task_additive_noise",
    "task_bivariate_normal",
    "task_bimodal_gaussians",
    "task_multinormal_lvm",
    "task_multinormal_dense",
    "task_multinormal_sparse",
    "task_multinormal_2pair",
    "task_x",
    "task_ai",
    "task_waves",
    "task_galaxy",
    "task_concentric_multinormal",
    "task_multinormal_sparse_w_inliers",
    "task_student_dense",
    "task_student_sparse",
    "task_student_2pair",
    "transform_wiggly_task",
    "transform_spiral_task",
    "transform_rotate_task",
    "transform_normal_cdf_task",
    "transform_swissroll_task",
    "transform_half_cube_task",
    "transform_gaussianise",
    "transform_uniformise",
    "transform_rescale",
]
