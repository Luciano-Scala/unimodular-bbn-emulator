"""
diagnose_prior.py
==================
Diagnóstico rápido: ¿qué fracción del espacio de prior sobrevive el nuevo
chequeo de suavidad (max_abs_eps2), y en qué rango de alpha/Nf?
"""
import numpy as np
from unimodular_physics import UnimodularModel
from mcmc_sampler import MODEL_SPECS, log_prior, initial_walkers

for model_name in ["tanh", "sinh2n"]:
    bounds = MODEL_SPECS[model_name]["bounds"]
    n_test = 2000
    rng = np.random.default_rng(0)

    # muestreo log-uniforme en alpha, uniforme en el resto (mismo que LHS)
    alpha_s = 10 ** rng.uniform(np.log10(bounds["alpha"][0]),
                                 np.log10(bounds["alpha"][1]), n_test)
    Nf_s = rng.uniform(bounds["Nf"][0], bounds["Nf"][1], n_test)
    if model_name == "sinh2n":
        n_s = rng.uniform(bounds["n"][0], bounds["n"][1], n_test)

    n_pass_min_eps1 = 0
    n_pass_eps2 = 0
    n_pass_both = 0
    eps2_values = []

    for i in range(n_test):
        params = {"alpha": alpha_s[i], "Nf": Nf_s[i]}
        if model_name == "sinh2n":
            params["n"] = n_s[i]
        try:
            model = UnimodularModel(model_name, params)
            if not model.is_valid:
                continue
            min_eps1_ok = model.min_eps1_during_inflation() <= 0.05
            eps2_val = model.max_abs_eps2_during_inflation()
            eps2_values.append(eps2_val)
            eps2_ok = eps2_val <= 0.3
            if min_eps1_ok:
                n_pass_min_eps1 += 1
            if eps2_ok:
                n_pass_eps2 += 1
            if min_eps1_ok and eps2_ok:
                n_pass_both += 1
        except Exception:
            continue

    print(f"\n=== {model_name} ({n_test} muestras) ===")
    print(f"Pasan min(eps1)<=0.05        : {n_pass_min_eps1} ({100*n_pass_min_eps1/n_test:.1f}%)")
    print(f"Pasan max|eps2|<=0.3         : {n_pass_eps2} ({100*n_pass_eps2/n_test:.1f}%)")
    print(f"Pasan AMBOS                  : {n_pass_both} ({100*n_pass_both/n_test:.1f}%)")
    if eps2_values:
        eps2_arr = np.array(eps2_values)
        print(f"max|eps2| -> min={eps2_arr.min():.3g}, mediana={np.median(eps2_arr):.3g}, "
              f"max={eps2_arr.max():.3g}")