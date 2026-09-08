"""
n_sensitivity_fine.py (v2, con gráfico)
=========================================
"""
import numpy as np
import matplotlib.pyplot as plt
from n_sensitivity_scan import fractions_for_n

if __name__ == "__main__":
    n_values = np.linspace(0.20, 0.90, 49)
    frac_bbn_list, frac_joint_list = [], []

    for n_val in n_values:
        fb, fj = fractions_for_n(n_val, n_grid=150)
        frac_bbn_list.append(fb)
        frac_joint_list.append(fj)
        print(f"n={n_val:.3f}  frac_bbn={fb*100:.2f}%  frac_joint={fj*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, np.array(frac_bbn_list) * 100, "o-", label="frac_bbn", alpha=0.6)
    ax.plot(n_values, np.array(frac_joint_list) * 100, "s-", label="frac_joint", color="C1")
    ax.axvline(0.4789, color="red", linestyle="--", alpha=0.5, label="MAP MCMC (n=0.479)")
    ax.set_xlabel("n")
    ax.set_ylabel("Fracción de la grilla (%)")
    ax.set_title(r"Barrido fino en $n$: región conjunta BBN($N_{\rm eff}$)+$\Lambda_{\rm obs}$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("n_sensitivity_fine.png", dpi=150, bbox_inches="tight")
    print("\nGuardado: n_sensitivity_fine.png")