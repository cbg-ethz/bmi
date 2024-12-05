import time

import bmi.benchmark.tasks.mixtures as mix


def evaluate_task(factory, samples: int):
    t0 = time.time()

    task = factory(samples)

    print(f"{task.mutual_information:.4f}  +- {task.sampler.mutual_information_std():.4f}")

    t1 = time.time()
    print(f"{t1 - t0:.2f}")


def main():
    TASK_FACTORIES = {
        "Galaxy": lambda n: mix.task_galaxy(mi_estimate_sample=n),
        "Concentric-25": lambda n: mix.task_concentric_multinormal(
            dim_x=25, n_components=5, mi_estimate_sample=n
        ),
    }
    MC_SAMPLES = 200_000

    for task_name, task_factory in TASK_FACTORIES.items():
        print(task_name)
        evaluate_task(task_factory, samples=MC_SAMPLES)


if __name__ == "__main__":
    main()
