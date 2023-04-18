# Snakemake workflow for experiments with estimating mutual information
# using a spiral distribution with varying speed
import resource

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import bmi

N_POINTS: int = 1_000
CORRELATION: float = 0.8
N_SEED: int = 3
SPEEDS: list[float] = [0, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3]


ESTIMATORS = {
    # TODO(Pawel): Add more estimators
    'KSG-5': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'CCA': bmi.estimators.CCAMutualInformationEstimator(),
}
# Estimator names on the generated figure
ESTIMATOR_NAMES = {
    'KSG-5': "KSG",
    'CCA': "CCA",
}

assert tuple(sorted(ESTIMATOR_NAMES.keys())) == tuple(sorted(ESTIMATORS.keys())), "Estimator IDs don't agree."


def spiral_task(speed: float, correlation) -> bmi.benchmark.Task:
    # Correlation cor(X1, Y) is non-zero.
    # We have cor(X1, X2) = cor(X2, Y) = 0.
    covariance = np.eye(3)
    covariance[0, 2] = CORRELATION
    covariance[2, 0] = CORRELATION

    generator = bmi.transforms.so_generator(2,0,1)

    sampler = bmi.samplers.TransformedSampler(
        base_sampler=bmi.samplers.SplitMultinormal(dim_x=2,dim_y=1,covariance=covariance),
        transform_x=bmi.transforms.Spiral(generator=generator,speed=speed),
    )
    return bmi.benchmark.Task(
        sampler=sampler,
        task_params={
            "correlation": correlation,
            "speed": speed,
        },
        task_name=f"Speed {speed:.2f}",
        task_id=f"spiral-{correlation}-{speed}",
    )


rule all:
    input:
        expand(
            'generated/spiral/figures/{n_samples}-{correlation}.pdf',
            n_samples=[N_POINTS],
            correlation=[CORRELATION],
        )

rule plot:
    input: 'generated/spiral/results_csv/{n_samples}-{correlation}.csv'
    output: 'generated/spiral/figures/{n_samples}-{correlation}.pdf'
    run:
        df = pd.read_csv(str(input))
        mi_true = spiral_task(speed=0, correlation=float(wildcards.correlation)).mutual_information

        fig, ax = plt.subplots()

        # Plot true mutual information
        ax.hlines([mi_true], min(SPEEDS), max(SPEEDS), colors="k",linestyles="--")

        for estimator_id, group in df.groupby("estimator_id"):
            sns.regplot(
                data=group,
                x="speed",
                y="mi_estimate",
                ax=ax,
                x_estimator=np.mean,
                lowess=True,
                label=ESTIMATOR_NAMES[estimator_id],
            )

        ax.set_xlabel("Speed")
        ax.set_ylabel("Mutual Information estimate")

        ax.legend()

        fig.tight_layout()
        fig.savefig(str(output), dpi=350)


rule assemble_results:
    # Assembles results from YAML files into one CSV
    output: 'generated/spiral/results_csv/{n_samples}-{correlation}.csv'
    input:
        expand(
            'generated/spiral/results/{estimator_id}-{n_samples}-{seed}-{correlation}-{speed}.yaml',
            estimator_id=ESTIMATORS,
            seed=range(N_SEED),
            speed=SPEEDS,
            n_samples=[N_POINTS],
            correlation=[CORRELATION],
        )
    run:
        def flatten_dict(d, sep='_'):
            items = []
            for key, v in d.items():
                if isinstance(v,dict):
                    items.extend(flatten_dict(v, sep=sep).items())
                else:
                    items.append((key, v))
            return dict(items)

        entries = []
        for input_name in input:
            with open(input_name) as handler:
                result = yaml.safe_load(handler)
                result = flatten_dict(result)

                task = spiral_task(speed=result["speed"], correlation=result["correlation"])
                result["true_mi"] = task.mutual_information

                entries.append(result)

        pd.DataFrame(entries).to_csv(str(output), index=False)


rule sample_task:
    # Sample a specified task and save to CSV
    output: 'generated/spiral/samples/{n_samples}-{seed}-{correlation}-{speed}.csv'
    run:
        n_samples = int(wildcards.n_samples)
        seed = int(wildcards.seed)
        correlation = float(wildcards.correlation)
        speed = float(wildcards.speed)

        task = spiral_task(speed=speed, correlation=correlation)
        task.save_sample(
            str(output),
            n_samples=n_samples,
            seed=seed,
        )


rule apply_estimator:
    # Apply estimator to a given sample and save result
    output: 'generated/spiral/results/{estimator_id}-{n_samples}-{seed}-{correlation}-{speed}.yaml'
    input: 'generated/spiral/samples/{n_samples}-{seed}-{correlation}-{speed}.csv'
    run:
        correlation = float(wildcards.correlation)
        n_samples = int(wildcards.n_samples)
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS[estimator_id]
        speed = float(wildcards.speed)
        seed = int(wildcards.seed)

        # this should be about ~4GiB
        resource.setrlimit(resource.RLIMIT_AS, (1 << 33, 1 << 33))

        result = bmi.benchmark.run_estimator(
            estimator=estimator,
            estimator_id=estimator_id,
            sample_path=str(input),
            task_id=f"spiral-{n_samples}-{seed}-{correlation}-{speed}",
            seed=seed,
            additional_information={
                "n_samples": n_samples,
                "speed": speed,
                "correlation": correlation,
            }
        )
        result.dump(str(output))
