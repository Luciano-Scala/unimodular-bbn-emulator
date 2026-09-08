"""
validate_chi2_emulator.py (v2)
================================
Actualizado para el nuevo target del emulador (dNeff_fo, dNeff_db) y el
nuevo chi2 basado en Neff directo (Goldstein & Hill 2026).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcmc_sampler import MODEL_SPECS
from bbn_evaluator import DNEFF_OBS, DNEFF_OBS_ERR
from emulator_model import BBNEmulator


def compute_chi2(dNeff_fo):
    return ((dNeff_fo - DNEFF_OBS) / DNEFF_OBS_ERR) ** 2


def validate(model_name, csv_path, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = f"emulator_{model_name}.pt"

    emu = BBNEmulator(checkpoint_path)
    df = pd.read_csv(csv_path)
    df = df[df["valid"] == True].reset_index(drop=True)
    names = list(MODEL_SPECS[model_name]["bounds"].keys())

    chi2_true = compute_chi2(df["dNeff_fo"].values)  # chi2 exacto, recalculado con la fórmula nueva
    chi2_emu = np.empty(len(df))
    dNeff_fo_emu = np.empty(len(df))

    for i, row in df.iterrows():
        theta = row[names].values.astype(float)
        dNeff_fo_pred, _ = emu.predict(theta)
        dNeff_fo_emu[i] = dNeff_fo_pred
        chi2_emu[i] = compute_chi2(dNeff_fo_pred)

    abs_err_chi2 = np.abs(chi2_emu - chi2_true)

    print(f"\n=== Validación de chi2 (Neff directo) para '{model_name}' ({len(df)} puntos) ===")
    print(f"|Delta chi2| -> mediana={np.median(abs_err_chi2):.4f}  "
          f"p95={np.percentile(abs_err_chi2, 95):.4f}  max={np.max(abs_err_chi2):.4f}")

    LOW_CHI2_THRESHOLD = 10.0
    mask_low = chi2_true < LOW_CHI2_THRESHOLD
    if mask_low.sum() > 0:
        print(f"\nSubconjunto con chi2_true < {LOW_CHI2_THRESHOLD} ({mask_low.sum()} puntos):")
        print(f"|Delta chi2| -> mediana={np.median(abs_err_chi2[mask_low]):.4f}  "
              f"p95={np.percentile(abs_err_chi2[mask_low], 95):.4f}  "
              f"max={np.max(abs_err_chi2[mask_low]):.4f}")

    worst_idx = np.argsort(abs_err_chi2)[-10:][::-1]
    print("\nPeores 10 puntos:")
    for idx in worst_idx:
        params_str = ", ".join(f"{n}={df.iloc[idx][n]:.4g}" for n in names)
        print(f"  {params_str} | chi2_true={chi2_true[idx]:.3f} chi2_emu={chi2_emu[idx]:.3f} | "
              f"dNeff_fo_true={df.iloc[idx]['dNeff_fo']:.4e} dNeff_fo_emu={dNeff_fo_emu[idx]:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(chi2_true, chi2_emu, s=5, alpha=0.3)
    lims = [0, min(50, chi2_true.max())]
    axes[0].plot(lims, lims, "r--", label="identidad")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("chi2 (física exacta)"); axes[0].set_ylabel("chi2 (emulador)")
    axes[0].set_title(f"'{model_name}': zoom en chi2 < 50"); axes[0].legend()
    axes[1].hist(abs_err_chi2, bins=60, log=True)
    axes[1].set_xlabel("|chi2_emu - chi2_true|"); axes[1].set_ylabel("cuenta (log)")
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