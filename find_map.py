"""
find_map.py
============
Extrae el MAP (Maximum A Posteriori) real de las cadenas guardadas en los
backends HDF5, en vez de usar medianas marginales 1D (que pueden no
corresponder a ningún punto conjunto de alta probabilidad si hay
correlaciones fuertes entre parámetros).

USO:
    python find_map.py --model tanh
    python find_map.py --model sinh2n
"""

import argparse
import numpy as np
import emcee

from mcmc_sampler import MODEL_SPECS
from unimodular_physics import UnimodularModel
from bbn_evaluator import bbn_chi2
from mcmc_sampler import LOG10_LAMBDA_OBS, SIGMA_LATE_DEX


def find_map(model_name, chain_file=None):
    if chain_file is None:
        chain_file = f"chain_{model_name}_emu.h5"

    backend = emcee.backends.HDFBackend(chain_file, read_only=True)
    n_iter = backend.iteration
    if n_iter == 0:
        print(f"[AVISO] '{chain_file}' no tiene pasos guardados.")
        return

    try:
        tau = backend.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
    except Exception:
        burnin = n_iter // 4

    # Sin thinning acá: queremos TODAS las muestras post burn-in para no
    # perder el punto de máxima probabilidad por submuestreo.
    chain = backend.get_chain(discard=burnin, flat=True)
    log_prob = backend.get_log_prob(discard=burnin, flat=True)

    finite_mask = np.isfinite(log_prob)
    chain = chain[finite_mask]
    log_prob = log_prob[finite_mask]

    idx_map = np.argmax(log_prob)
    theta_map = chain[idx_map]
    logpost_map = log_prob[idx_map]

    names = list(MODEL_SPECS[model_name]["bounds"].keys())
    labels = MODEL_SPECS[model_name]["labels"]

    print(f"\n=== MAP para '{model_name}' ({len(chain)} muestras post burn-in) ===")
    for name, label, val in zip(names, labels, theta_map):
        print(f"  {label} = {val:.4g}")
    print(f"  log_posterior(MAP) = {logpost_map:.4f}")

    # Comparación directa con medianas marginales 1D (lo que usamos antes)
    medians = np.median(chain, axis=0)
    print(f"\nComparación MAP vs. medianas marginales 1D:")
    for name, label, map_val, med_val in zip(names, labels, theta_map, medians):
        diff_pct = 100 * abs(map_val - med_val) / abs(med_val) if med_val != 0 else np.nan
        print(f"  {label}: MAP={map_val:.4g}  mediana={med_val:.4g}  "
              f"diff={diff_pct:.1f}%")

    # Física del punto MAP: chi2 de BBN y desviación de Lambda_obs
    params = dict(zip(names, theta_map))
    model = UnimodularModel(model_name, params)
    result = bbn_chi2(model)
    Q_late = model.late_time_Q_over_MP4_proxy()
    log10_Q_late = np.log10(Q_late) if Q_late > 0 else np.nan
    desviacion_dex = log10_Q_late - LOG10_LAMBDA_OBS

    print(f"\nFísica en el punto MAP:")
    print(f"  chi2_BBN total       = {result['chi2_total']:.2f}")
    print(f"  log10(Q_late/M_P^4)  = {log10_Q_late:.2f}  "
          f"(target: {LOG10_LAMBDA_OBS:.2f}, desviacion: {desviacion_dex:+.2f} dex, "
          f"prior sigma={SIGMA_LATE_DEX} dex)")

    return theta_map, result, desviacion_dex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    args = parser.parse_args()
    find_map(args.model)