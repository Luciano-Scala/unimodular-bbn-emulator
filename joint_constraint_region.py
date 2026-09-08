"""
joint_constraint_region.py
============================
Corrige el enfoque anterior (buscar un MAP puntual en una meseta plana)
por el correcto: identificar la region alpha > alpha_min permitida por
BBN, y DENTRO de esa region, mapear que sub-conjunto de (Nf, n) reproduce
Q_late ~ Lambda_obs dentro de una tolerancia fija (no un prior gaussiano
suave, sino una banda de aceptacion explicita p.ej. +-0.5 dex).
"""
import numpy as np
import matplotlib.pyplot as plt

from mcmc_sampler import MODEL_SPECS, LOG10_LAMBDA_OBS
from unimodular_physics import UnimodularModel
from emulator_model import BBNEmulator
# from bbn_evaluator import YP_OBS, YP_ERR, DH_OBS, DH_ERR
from bbn_evaluator import DNEFF_OBS, DNEFF_OBS_ERR

CHI2_THRESHOLD = 5.99   # ~95% CL, 2 dof
LATE_TIME_TOLERANCE_DEX = 0.5


# def compute_chi2_emu(emu, theta):
#     Yp_pred, DH_pred = emu.predict(theta)
#     chi2_Yp = ((Yp_pred - YP_OBS) / YP_ERR) ** 2
#     chi2_DH = ((DH_pred - DH_OBS) / DH_ERR) ** 2
#     return chi2_Yp + chi2_DH

def compute_chi2_emu(emu, theta):
    dNeff_fo_pred, dNeff_db_pred = emu.predict(theta)
    return ((dNeff_fo_pred - DNEFF_OBS) / DNEFF_OBS_ERR) ** 2

def scan(model_name, n_fixed=None, n_grid=200, output_suffix=""):
    from mcmc_sampler import passes_shape_prior 
    spec = MODEL_SPECS[model_name]
    bounds = spec["bounds"]
    emu = BBNEmulator(f"emulator_{model_name}.pt")

    alpha_grid = np.logspace(np.log10(bounds["alpha"][0]),
                              np.log10(bounds["alpha"][1]), n_grid)
    Nf_grid = np.linspace(bounds["Nf"][0], bounds["Nf"][1], n_grid)
    A, N = np.meshgrid(alpha_grid, Nf_grid)

    CHI2 = np.full_like(A, np.nan)
    DEX_DEV = np.full_like(A, np.nan)

    for i in range(n_grid):
        for j in range(n_grid):
            params = {"alpha": A[i, j], "Nf": N[i, j]}
            if model_name == "sinh2n":
                params["n"] = n_fixed
            theta = list(params.values())

            try:
                model = UnimodularModel(model_name, params)
                if not passes_shape_prior(model):
                    continue  # deja CHI2/DEX_DEV como NaN en este punto
                CHI2[i, j] = compute_chi2_emu(emu, theta)
                Q_late = model.late_time_Q_over_MP4_proxy()
                if Q_late > 0:
                    DEX_DEV[i, j] = np.log10(Q_late) - LOG10_LAMBDA_OBS
            except Exception:
                pass

    bbn_ok = CHI2 < CHI2_THRESHOLD
    late_ok = np.abs(DEX_DEV) < LATE_TIME_TOLERANCE_DEX
    joint_ok = bbn_ok & late_ok

    frac_bbn = np.nanmean(bbn_ok)
    frac_joint = np.nanmean(joint_ok)
    print(f"[{model_name}] Fraccion de la grilla con BBN OK (chi2<{CHI2_THRESHOLD}): "
          f"{frac_bbn*100:.1f}%")
    print(f"[{model_name}] Fraccion con BBN Y Lambda OK simultaneamente "
          f"(dentro de +-{LATE_TIME_TOLERANCE_DEX} dex): {frac_joint*100:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(A, N, bbn_ok.astype(float), levels=[0.5, 1.5],
                colors=["none"], hatches=[None])
    ax.contourf(A, N, np.where(bbn_ok, 1, np.nan), levels=[0.5, 1.5],
                colors=["#cce5ff"], alpha=0.5)
    ax.contourf(A, N, np.where(joint_ok, 1, np.nan), levels=[0.5, 1.5],
                colors=["#2ca02c"], alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$N_f$")
    title_extra = f", n={n_fixed:.3g} fijo" if model_name == "sinh2n" else ""
    ax.set_title(f"'{model_name}'{title_extra}\n"
                 f"Azul claro: BBN OK | Verde: BBN OK Y Lambda_obs OK (+-{LATE_TIME_TOLERANCE_DEX} dex)")

    out = f"joint_region_{model_name}{output_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out}")
    plt.close(fig)


if __name__ == "__main__":
    scan("tanh",None,200,"")
    for n_val in [1.0, 2.5, 5.0]:
        scan("sinh2n", n_fixed=n_val,n_grid=200,output_suffix=n_val)