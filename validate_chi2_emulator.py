"""
validate_chi2_emulator.py
==========================
Valida el emulador NO en el espacio (Yp, D/H) directamente, sino en el
espacio que realmente usa el MCMC: chi2_total. Esto es más estricto que
el criterio de error relativo en Yp/D-H, porque DH_ERR es muy chico
(~1.6% relativo) y por lo tanto pequeños errores del emulador en D/H
pueden traducirse en errores GRANDES de chi2 -- justo donde el sampler
pasa la mayor parte del tiempo (cerca del modo del posterior).

USO:
    python validate_chi2_emulator.py --model tanh   --csv dataset_tanh.csv
    python validate_chi2_emulator.py --model sinh2n --csv dataset_sinh2n.csv
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcmc_sampler import MODEL_SPECS
from bbn_evaluator import YP_OBS, YP_ERR, DH_OBS, DH_ERR
from emulator_model import BBNEmulator


def compute_chi2(Yp, DH):
    chi2_Yp = ((Yp - YP_OBS) / YP_ERR) ** 2
    chi2_DH = ((DH - DH_OBS) / DH_ERR) ** 2
    return chi2_Yp + chi2_DH


def validate(model_name, csv_path, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = f"emulator_{model_name}.pt"

    emu = BBNEmulator(checkpoint_path)
    df = pd.read_csv(csv_path)
    df = df[df["valid"] == True].reset_index(drop=True)

    names = list(MODEL_SPECS[model_name]["bounds"].keys())

    chi2_true = df["chi2_total"].values.copy()
    chi2_emu = np.empty(len(df))
    Yp_emu = np.empty(len(df))
    DH_emu = np.empty(len(df))

    for i, row in df.iterrows():
        theta = row[names].values.astype(float)
        Yp_pred, DH_pred = emu.predict(theta)
        Yp_emu[i] = Yp_pred
        DH_emu[i] = DH_pred
        chi2_emu[i] = compute_chi2(Yp_pred, DH_pred)

    abs_err_chi2 = np.abs(chi2_emu - chi2_true)

    print(f"\n=== Validación de chi2 para '{model_name}' ({len(df)} puntos) ===")
    print(f"|Delta chi2| -> mediana={np.median(abs_err_chi2):.4f}  "
          f"p95={np.percentile(abs_err_chi2, 95):.4f}  "
          f"max={np.max(abs_err_chi2):.4f}")

    # Lo que realmente importa: el error de chi2 CERCA del modo del
    # posterior (chi2_true bajo), no en la cola ya descartada.
    LOW_CHI2_THRESHOLD = 10.0
    mask_low = chi2_true < LOW_CHI2_THRESHOLD
    n_low = mask_low.sum()
    if n_low > 0:
        print(f"\nSubconjunto con chi2_true < {LOW_CHI2_THRESHOLD} "
              f"({n_low} puntos, la región relevante para el posterior):")
        print(f"|Delta chi2| -> mediana={np.median(abs_err_chi2[mask_low]):.4f}  "
              f"p95={np.percentile(abs_err_chi2[mask_low], 95):.4f}  "
              f"max={np.max(abs_err_chi2[mask_low]):.4f}")
    else:
        print(f"\n[AVISO] Ningún punto del dataset tiene chi2_true < "
              f"{LOW_CHI2_THRESHOLD}. No podemos evaluar el error del "
              f"emulador en la región que le importa al posterior con "
              f"este dataset -- el LHS no muestreó esa zona (esperable si "
              f"la tensión con BBN es fuerte en toda la región del prior).")

    # Diagnóstico: ¿dónde están los peores errores?
    worst_idx = np.argsort(abs_err_chi2)[-10:][::-1]
    print("\nPeores 10 puntos (parámetros y chi2 true vs emulado):")
    for idx in worst_idx:
        params_str = ", ".join(f"{n}={df.iloc[idx][n]:.4g}" for n in names)
        print(f"  {params_str} | chi2_true={chi2_true[idx]:.3f} "
              f"chi2_emu={chi2_emu[idx]:.3f} | "
              f"DH_true={df.iloc[idx]['DH_pred']:.4e} DH_emu={DH_emu[idx]:.4e}")

    # Plot: dispersión chi2_true vs chi2_emu, coloreado por si están en
    # la región de bajo chi2 o no
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(chi2_true, chi2_emu, s=5, alpha=0.3)
    lims = [0, min(50, chi2_true.max())]
    axes[0].plot(lims, lims, "r--", label="identidad")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("chi2 (física exacta)")
    axes[0].set_ylabel("chi2 (emulador)")
    axes[0].set_title(f"'{model_name}': zoom en chi2 < 50")
    axes[0].legend()

    axes[1].hist(abs_err_chi2, bins=60, log=True)
    axes[1].set_xlabel("|chi2_emu - chi2_true|")
    axes[1].set_ylabel("cuenta (log)")
    axes[1].set_title("Distribución del error de chi2")

    fig.tight_layout()
    out_png = f"chi2_validation_{model_name}.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nGráfico guardado en: {out_png}")

    return chi2_true, chi2_emu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    validate(args.model, args.csv)