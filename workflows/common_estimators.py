import bmi.estimators as estimators
import bmi.estimators.external.julia_estimators as julia_estimators
import bmi.estimators.external.r_estimators as r_estimators

ESTIMATORS = {
    "MINE": estimators.MINEEstimator(verbose=False),
    "InfoNCE": estimators.InfoNCEEstimator(verbose=False),
    "NWJ": estimators.NWJEstimator(verbose=False),
    "Donsker-Varadhan": estimators.DonskerVaradhanEstimator(verbose=False),
    # 'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    # 'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    # 'Hist-10': estimators.HistogramEstimator(n_bins_x=10),
    "R-KSG-I-5": r_estimators.RKSGEstimator(variant=1, neighbors=5),
    # 'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
    # 'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
    # 'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
    # 'R-BNSL': r_estimators.RBNSLEstimator(),
    "R-LNN": r_estimators.RLNNEstimator(),
    "Julia-Hist-10": julia_estimators.JuliaHistogramEstimator(bins=10),
    # 'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
    # 'Julia-Transfer-30': julia_estimators.JuliaTransferEstimator(bins=30),
    # 'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
}

ESTIMATOR_COLORS = {
    "MINE": "purple",
    "InfoNCE": "magenta",
    "Donsker-Varadhan": "red",
    "NWJ": "orangered",
    "R-KSG-I-5": "mediumblue",
    "R-LNN": "goldenrod",
    "Julia-Hist-10": "limegreen",
}

# shown on plots
ESTIMATOR_NAMES = {
    "MINE": "MINE",
    "InfoNCE": "InfoNCE",
    "Donsker-Varadhan": "D-V",
    "NWJ": "NWJ",
    "R-KSG-I-5": "KSG I (R)",
    "R-LNN": "LNN (R)",
    "Julia-Hist-10": "Hist. (Julia)",
}


assert set(ESTIMATORS.keys()) == set(ESTIMATOR_COLORS.keys())
assert set(ESTIMATORS.keys()) == set(ESTIMATOR_NAMES.keys())
