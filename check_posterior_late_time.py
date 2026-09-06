"""
check_posterior_late_time.py
==============================
Chequea si el punto de máxima probabilidad posterior (BBN-compatible)
también reproduce el orden de magnitud de la CC observada hoy, o si el
ajuste de BBN se logró sacrificando ese objetivo original del paper.
"""
import numpy as np
from unimodular_physics import UnimodularModel
from mcmc_sampler import LOG10_LAMBDA_OBS, SIGMA_LATE_DEX
from bbn_evaluator import bbn_chi2

candidates = {
    "tanh (paper original)":  ("tanh",   {"alpha": 0.027, "Nf": 300.0}),
    "tanh (posterior BBN)":   ("tanh",   {"alpha": 0.444, "Nf": 207.6}),
    "sinh2n (posterior BBN)": ("sinh2n", {"alpha": 0.558, "Nf": 231.3, "n": 3.036}),
}

print(f"Objetivo: log10(Lambda_obs/M_P^2) = {LOG10_LAMBDA_OBS:.2f} "
      f"(prior suave sigma={SIGMA_LATE_DEX} dex)\n")

for name, (model_name, params) in candidates.items():
    model = UnimodularModel(model_name, params)
    Q_late = model.late_time_Q_over_MP4_proxy()
    log10_Q_late = np.log10(Q_late) if Q_late > 0 else np.nan
    desviacion_dex = log10_Q_late - LOG10_LAMBDA_OBS

    result = bbn_chi2(model)

    print(f"{name}:")
    print(f"  chi2_BBN total       = {result['chi2_total']:.2f}")
    print(f"  log10(Q_late/M_P^4)  = {log10_Q_late:.2f}  "
          f"(target: {LOG10_LAMBDA_OBS:.2f}, desviacion: {desviacion_dex:+.2f} dex)")
    print()