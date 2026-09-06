"""
n_sensitivity_scan.py
=======================
Barrido fino en n (no solo 3 puntos) de frac_bbn y frac_joint, para
caracterizar si la caida de la fraccion conjunta es monotona/suave o
si hay un umbral abrupto en algun n intermedio.
"""
import numpy as np
import matplotlib.pyplot as plt

from joint_constraint_region import scan
from mcmc_sampler import MODEL_SPECS
from emulator_model import BBNEmulator
from unimodular_physics import UnimodularModel
from mcmc_sampler import LOG10_LAMBDA_OBS
from mcmc_sampler import passes_shape_prior
from bbn_evaluator import YP_OBS, YP_ERR, DH_OBS, DH_ERR

CHI2_THRESHOLD = 5.99
LATE_TIME_TOLERANCE_DEX = 0.5


def compute_chi2_emu(emu, theta):
    Yp_pred, DH_pred = emu.predict(theta)
    chi2_Yp = ((Yp_pred - YP_OBS) / YP_ERR) ** 2
    chi2_DH = ((DH_pred - DH_OBS) / DH_ERR) ** 2
    return chi2_Yp + chi2_DH


def fractions_for_n(n_fixed, n_grid=120):
    bounds = MODEL_SPECS["sinh2n"]["bounds"]
    emu = BBNEmulator("emulator_sinh2n.pt")

    alpha_grid = np.logspace(np.log10(bounds["alpha"][0]),
                              np.log10(bounds["alpha"][1]), n_grid)
    Nf_grid = np.linspace(bounds["Nf"][0], bounds["Nf"][1], n_grid)
    A, N = np.meshgrid(alpha_grid, Nf_grid)

    CHI2 = np.full_like(A, np.nan)
    DEX_DEV = np.full_like(A, np.nan)

    for i in range(n_grid):
        for j in range(n_grid):
            params = {"alpha": A[i, j], "Nf": N[i, j], "n": n_fixed}
            theta = list(params.values())
            try:
                model = UnimodularModel("sinh2n", params)
                if not passes_shape_prior(model):
                    continue
                CHI2[i, j] = compute_chi2_emu(emu, theta)
                Q_late = model.late_time_Q_over_MP4_proxy()
                if Q_late > 0:
                    DEX_DEV[i, j] = np.log10(Q_late) - LOG10_LAMBDA_OBS
            except Exception:
                pass

    bbn_ok = CHI2 < CHI2_THRESHOLD
    late_ok = np.abs(DEX_DEV) < LATE_TIME_TOLERANCE_DEX
    joint_ok = bbn_ok & late_ok

    return np.nanmean(bbn_ok), np.nanmean(joint_ok)


if __name__ == "__main__":
    n_values = np.linspace(0.5, 6.0, 20)  # cubre todo el rango del prior
    frac_bbn_list, frac_joint_list = [], []

    for n_val in n_values:
        fb, fj = fractions_for_n(n_val)
        frac_bbn_list.append(fb)
        frac_joint_list.append(fj)
        print(f"n={n_val:.2f}  frac_bbn={fb*100:.1f}%  frac_joint={fj*100:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, np.array(frac_bbn_list) * 100, "o-", label="frac_bbn")
    ax.plot(n_values, np.array(frac_joint_list) * 100, "s-", label="frac_joint")
    ax.set_xlabel("n")
    ax.set_ylabel("Fracción de la grilla (%)")
    ax.set_title("Sensibilidad a n: compatibilidad BBN vs. BBN+Lambda conjunta")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("n_sensitivity.png", dpi=150, bbox_inches="tight")
    print("\nGuardado: n_sensitivity.png")