"""
plot_degeneracy.py
====================
Diagnostica la degenerescencia observada en el posterior de 'sinh2n'
(y, en menor medida, la relación alpha-Nf de 'tanh'):

  1. Corner plot de la cadena MCMC completa (más allá de cuantiles 1D).
  2. Barrido explícito de chi2_BBN en la grilla (alpha, Nf), fijando n
     en su valor de MAP, usando el emulador (rápido). Esto muestra la
     FORMA geométrica de la cresta de baja chi2: si es una curva angosta
     y continua, es degenerescencia real (los datos de BBN solo fijan
     una combinación de alpha y Nf, no cada uno por separado). Si en
     cambio aparecen varias islas separadas por regiones de chi2 alto,
     es multimodalidad genuina.

USO:
    python plot_degeneracy.py --model sinh2n
    python plot_degeneracy.py --model tanh
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

from mcmc_sampler import MODEL_SPECS
from emulator_model import BBNEmulator
from bbn_evaluator import YP_OBS, YP_ERR, DH_OBS, DH_ERR


def compute_chi2_emu(emu, theta):
    Yp_pred, DH_pred = emu.predict(theta)
    chi2_Yp = ((Yp_pred - YP_OBS) / YP_ERR) ** 2
    chi2_DH = ((DH_pred - DH_OBS) / DH_ERR) ** 2
    return chi2_Yp + chi2_DH


def full_corner(model_name):
    """Corner plot de la cadena completa (todas las dimensiones a la vez)."""
    chain_file = f"chain_{model_name}_emu.h5"
    backend = emcee.backends.HDFBackend(chain_file, read_only=True)

    try:
        tau = backend.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
    except Exception:
        burnin, thin = backend.iteration // 4, 1

    samples = backend.get_chain(discard=burnin, thin=thin, flat=True)
    labels = MODEL_SPECS[model_name]["labels"]

    fig = corner.corner(
        samples, labels=labels, show_titles=True,
        quantiles=[0.16, 0.5, 0.84], title_fmt=".3g",
        levels=(0.393, 0.865, 0.988),  # ~1,2,3 sigma en 2D
    )
    fig.suptitle(f"Posterior completo (post burn-in) — '{model_name}'", y=1.02)
    out = f"corner_full_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Corner plot completo guardado en: {out}")
    plt.close(fig)

    return samples


def ridge_scan(model_name, n_fixed=None, n_grid=150):
    """
    Barrido de chi2_BBN(alpha, Nf) en grilla, con n fijo (solo aplica a
    sinh2n; para tanh, n_fixed se ignora y es un barrido 2D directo).
    """
    spec = MODEL_SPECS[model_name]
    bounds = spec["bounds"]
    emu = BBNEmulator(f"emulator_{model_name}.pt")

    alpha_lo, alpha_hi = bounds["alpha"]
    Nf_lo, Nf_hi = bounds["Nf"]

    alpha_grid = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), n_grid)
    Nf_grid = np.linspace(Nf_lo, Nf_hi, n_grid)
    A, N = np.meshgrid(alpha_grid, Nf_grid)
    CHI2 = np.full_like(A, np.nan)

    for i in range(n_grid):
        for j in range(n_grid):
            if model_name == "sinh2n":
                theta = [A[i, j], N[i, j], n_fixed]
            else:
                theta = [A[i, j], N[i, j]]
            try:
                CHI2[i, j] = compute_chi2_emu(emu, theta)
            except Exception:
                CHI2[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = [1, 2, 4, 9, 16, 25, 50, 100, 500]
    cs = ax.contourf(A, N, np.clip(CHI2, 0, 500), levels=levels,
                      cmap="viridis_r", extend="max")
    ax.contour(A, N, CHI2, levels=[5.99], colors="red", linewidths=2,
               linestyles="--")  # ~95% CL para 2 dof
    fig.colorbar(cs, label=r"$\chi^2_{BBN}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$N_f$")
    title_extra = f", n={n_fixed:.3g} fijo" if model_name == "sinh2n" else ""
    ax.set_title(f"Cresta de degenerescencia — '{model_name}'{title_extra}\n"
                 f"(línea roja: contorno chi2=5.99, ~95% CL con 2 gl)")

    out = f"ridge_{model_name}_1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Mapa de la cresta guardado en: {out}")
    plt.close(fig)

    return A, N, CHI2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--n_fixed", type=float, default=None,
                         help="Valor de n a fijar para el barrido (solo sinh2n). "
                              "Default: usa el n del MAP más reciente.")
    args = parser.parse_args()

    samples = full_corner(args.model)

    n_fixed = args.n_fixed
    if args.model == "sinh2n" and n_fixed is None:
        # Usa el n del MAP de la cadena (columna 2)
        n_fixed = float(np.median(samples[:, 2]))
        print(f"Usando n_fixed={n_fixed:.4g} (mediana de la cadena)")

    ridge_scan(args.model, n_fixed=1)