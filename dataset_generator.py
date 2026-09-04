"""
dataset_generator.py
=====================
Genera datasets de entrenamiento para el emulador de Fase 3, muestreando
el espacio de parámetros de cada familia de Q (tanh, sinh2n) con Latin
Hypercube Sampling y evaluando la física exacta (Fase 1) en cada punto.

Estrategia:
    - Muestreo LHS dentro de los mismos `bounds` usados en mcmc_sampler.py
      (fuente única de verdad para los priors: MODEL_SPECS).
    - Se filtra cada muestra con el mismo `log_prior` del MCMC: si no es
      finito, el punto es físicamente inválido (no hay inflación real,
      eps1 sale de (0,2), o no llega a diluirse hasta BBN dentro de N_max).
    - Se generan tandas hasta acumular `n_target_valid` puntos válidos.
    - Salida: un CSV por modelo, con columnas de parámetros + todo lo que
      devuelve bbn_chi2 (incluye Yp, D/H, que son el target del emulador).

USO:
    python dataset_generator.py --model tanh    --n_valid 10000
    python dataset_generator.py --model sinh2n  --n_valid 10000
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm

from unimodular_physics import UnimodularModel
from bbn_evaluator import bbn_chi2
from mcmc_sampler import MODEL_SPECS, log_prior


# ----------------------------------------------------------------------
def _lhs_unit_to_physical(unit_sample, model_name):
    """
    Convierte una muestra en el hipercubo unitario [0,1]^d a los valores
    físicos de los parámetros, respetando el prior log-uniforme en alpha
    (MODEL_SPECS["log_param"]).
    """
    spec = MODEL_SPECS[model_name]
    names = list(spec["bounds"].keys())
    theta = np.empty(len(names))

    for i, name in enumerate(names):
        lo, hi = spec["bounds"][name]
        u = unit_sample[i]
        if spec["log_param"].get(name, False):
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            theta[i] = 10 ** (log_lo + u * (log_hi - log_lo))
        else:
            theta[i] = lo + u * (hi - lo)
    return theta


def _evaluate_point(model_name, theta, ombh2):
    """
    Evalúa un único punto del espacio de parámetros: chequea el prior,
    y si es finito corre la física de BBN. Nunca lanza excepción hacia
    afuera (todo error se traduce en valid=False).
    """
    spec = MODEL_SPECS[model_name]
    names = list(spec["bounds"].keys())
    params = dict(zip(names, theta))

    row = dict(params)
    row["valid"] = False
    row["Gamma_fo"] = np.nan
    row["Gamma_db"] = np.nan
    row["dNeff_fo"] = np.nan
    row["dNeff_db"] = np.nan
    row["Yp_pred"] = np.nan
    row["DH_pred"] = np.nan
    row["chi2_total"] = np.nan

    lp = log_prior(theta, model_name)
    if not np.isfinite(lp):
        return row  # inválido por prior (incluye chequeos físicos de UnimodularModel)

    try:
        model = UnimodularModel(model_name, params)
        result = bbn_chi2(model, ombh2=ombh2)
    except Exception:
        return row  # inválido por falla numérica (root-finding, etc.)

    row["valid"] = True
    row["Gamma_fo"] = result["Gamma_fo"]
    row["Gamma_db"] = result["Gamma_db"]
    row["dNeff_fo"] = result["dNeff_fo"]
    row["dNeff_db"] = result["dNeff_db"]
    row["Yp_pred"] = result["Yp_pred"]
    row["DH_pred"] = result["DH_pred"]
    row["chi2_total"] = result["chi2_total"]
    return row


# ----------------------------------------------------------------------
def generate_dataset(model_name, n_valid_target=10_000, batch_size=2000,
                      ombh2=0.02237, seed=0, max_batches=200,
                      output_csv=None):
    """
    Genera muestras LHS en tandas hasta juntar n_valid_target puntos
    físicamente válidos. Devuelve un DataFrame con TODOS los puntos
    evaluados (válidos e inválidos, marcados con la columna 'valid'),
    para poder diagnosticar qué fracción del hipercubo es viable.
    """
    ndim = MODEL_SPECS[model_name]["ndim"]
    sampler = qmc.LatinHypercube(d=ndim, seed=seed)

    all_rows = []
    n_valid = 0
    batch_idx = 0
    t0 = time.time()

    pbar = tqdm(total=n_valid_target, desc=f"[{model_name}] puntos válidos")

    while n_valid < n_valid_target and batch_idx < max_batches:
        unit_batch = sampler.random(n=batch_size)
        for u in unit_batch:
            theta = _lhs_unit_to_physical(u, model_name)
            row = _evaluate_point(model_name, theta, ombh2)
            all_rows.append(row)
            if row["valid"]:
                n_valid += 1
                pbar.update(1)
            if n_valid >= n_valid_target:
                break
        batch_idx += 1

    pbar.close()

    df = pd.DataFrame(all_rows)
    elapsed = time.time() - t0
    frac_valid = df["valid"].mean()

    print(f"\n[{model_name}] Total evaluado: {len(df)} puntos "
          f"({n_valid} válidos, {frac_valid*100:.1f}% de aceptación) "
          f"en {elapsed:.1f} s ({elapsed/len(df)*1000:.2f} ms/punto)")

    if batch_idx >= max_batches and n_valid < n_valid_target:
        print(f"[AVISO] Se alcanzó max_batches={max_batches} sin llegar a "
              f"n_valid_target={n_valid_target}. Considerá angostar los "
              f"bounds en MODEL_SPECS o subir max_batches.")

    if output_csv is None:
        output_csv = f"dataset_{model_name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Dataset guardado en: {output_csv}")

    return df


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--n_valid", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_dataset(
        model_name=args.model,
        n_valid_target=args.n_valid,
        batch_size=args.batch_size,
        seed=args.seed,
    )