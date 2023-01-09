"""We have
    MI_student(Omega) = MI_Gaussian(Omega) + correction,
where the correction term depends on:
  - the dimensions of the variables
  - the degrees of freedom.

This script plots the correction term as a function
of the degrees of freedom for several dimensions.
"""
import matplotlib.pyplot as plt

import bmi.api as bmi


def correction(df: int, m: int, n: int) -> float:
    return bmi.samplers.SplitStudentT.mi_correction_function(df=df, dim_x=m, dim_y=n)


MARKER_LIST = [
    "o",
    "s",
    "x",
    "s",
    "v",
    "^",
    "P",
    "X",
    "1",
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    nus = list(range(1, 30))
    mns = [(1, 1), (1, 2), (1, 10), (2, 2), (3, 3), (5, 5), (10, 10)]

    for i, (m, n) in enumerate(mns):
        values = [correction(df=df, m=m, n=n) for df in nus]
        ax.scatter(nus, values, label=f"$m={m}$,\t$n={n}$", s=4, marker=MARKER_LIST[i])

    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("MI correction term")

    ax.legend()

    fig.tight_layout()
    fig.savefig("student-t-correction-plot.pdf")


if __name__ == "__main__":
    main()
