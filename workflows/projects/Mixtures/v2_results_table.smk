import numpy as np
import pandas as pd
import yaml


def read_results(
    path: str,
    unpack_task_params: bool = True,
    unpack_additional_information: bool = False,
):
    results = pd.read_csv(path)

    # read dicts
    for col in ["task_params", "additional_information"]:
        results[col] = results[col].apply(lambda x: yaml.safe_load(x))

    # unpack
    rows = []
    for _, row in results.iterrows():
        row_dict = row.to_dict()

        if unpack_task_params:
            row_dict |= row_dict["task_params"]
            del row_dict["task_params"]

        if unpack_additional_information:
            row_dict |= row_dict["additional_information"]
            del row_dict["additional_information"]

        rows.append(row_dict)

    results = pd.DataFrame(rows)
    return results



def create_benchmark_table(results, n_samples):
    if n_samples is None:
        n_samples = results["n_samples"].max()

    data = results[results["n_samples"] == n_samples]
    data = results[["estimator_id", "task_id", "n_samples", "mi_estimate", "mi_true"]].copy()

    # mean over seeds
    data_mean = (
        data.groupby(["estimator_id", "task_id"])["mi_estimate"]
        .mean()
        .reset_index()
    )
    data_std = (
        data.groupby(["estimator_id", "task_id"])["mi_estimate"]
        .std(ddof=1)
        .reset_index()
    )

    data_mean["mean"] = data_mean["mi_estimate"]
    data_std["std"] = data_std["mi_estimate"]

    merged = pd.merge(data_mean, data_std, on=['estimator_id', 'task_id'])
    merged["new_std"] = np.maximum(np.ceil(100 * merged["std"].values) / 100, 0.01)
    
    def format_output(row):
        mean = row["mean"]
        std = row["new_std"]
        return f"${mean:.2f} \\pm {std:.2f}$"
    
    merged["output"] = merged.apply(format_output, axis=1)
    merged = merged[["estimator_id", "task_id", "output"]]

    # add "True MI" pseudo-estimator
    task_id_mi_true = results[["task_id", "mi_true"]].drop_duplicates(subset=["task_id"])
    def make_mi_true_nice(x: float) -> str:
        y = f"{x:.2f}"
        return "\\textbf{" + y + "}"
    
    data = pd.concat(
        [
            merged,
            pd.DataFrame(
                {
                    "estimator_id": "True MI",
                    "task_id": task_id_mi_true["task_id"],
                    "output": task_id_mi_true['mi_true'].apply(make_mi_true_nice),
                }
            ),
        ]
    )

    return data.pivot(index="task_id", columns="estimator_id", values="output")


rule all:
    input: "generated/projects/Mixtures/benchmark_table.txt"


rule generate_table:
    input: "generated/benchmark/v2/results.csv"
    output: "generated/projects/Mixtures/benchmark_table.txt"
    run:
        input_df = read_results(str(input))
        # Use 5,000 samples
        output_df = create_benchmark_table(input_df, n_samples=5_000)

        names = {
            # one-dimensional
            'asinh-student-identity-1-1-1': "Student (1-dim)",
            '1v1-additive-0.75': "Additive",
            '1v1-AI': "AI",
            '1v1-X-0.9': "X",
            'swissroll_x-normal_cdf-1v1-normal-0.75': "Swiss roll",

            # m vs 1 dimension     
            '2v1-galaxy-0.5-3.0': "Galaxy",
            '2v1-waves-12-5.0-3.0': "Waves", 

            '25v1-concentric_gaussians-5': "Concentric (25-dim, 5)",
            '3v1-concentric_gaussians-10': "Concentric (3-dim, 10)",
            '3v1-concentric_gaussians-5': "Concentric (3-dim, 5)", 
            '5v1-concentric_gaussians-10': "Concentric (5-dim, 10)",
            '5v1-concentric_gaussians-5': "Concentric (5-dim, 5)", 

            # Multivariate normal
            'multinormal-dense-5-5-0.5': "Normal (5-dim, dense)",
            'multinormal-dense-25-25-0.5': "Normal (25-dim, dense)",
            'multinormal-dense-50-50-0.5': "Normal (50-dim, dense)",
            'multinormal-sparse-5-5-2-2.0': "Normal (5-dim, sparse)",
            'multinormal-sparse-25-25-2-2.0': "Normal (25-dim, sparse)",

            # Student
            'asinh-student-identity-2-2-1': "Student (2-dim)",
            'asinh-student-identity-3-3-2': "Student (3-dim)",
            'asinh-student-identity-5-5-2': "Student (5-dim)", 

            # Spiral
            'spiral-multinormal-sparse-3-3-2-2.0': "Spiral (3-dim)",
            'spiral-multinormal-sparse-5-5-2-2.0': "Spiral (5-dim)",

            # Inliers
            'mult-sparse-w-inliers-5-5-2-2.0-0.2': "Inliers (5-dim, 0.2)",
            'mult-sparse-w-inliers-5-5-2-2.0-0.5': "Inliers (5-dim, 0.5)",
            'mult-sparse-w-inliers-25-25-2-2.0-0.2': "Inliers (25-dim, 0.2)",
            'mult-sparse-w-inliers-25-25-2-2.0-0.5': "Inliers (25-dim, 0.5)",
        }

        output_df.index = output_df.index.map(lambda x: names[x])

        with open(str(output), "w") as fh:
            fh.write(output_df.to_latex(escape=False))
