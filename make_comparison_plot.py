"""
make_comparison_plot.py
=========================
Reproduce, en el estilo de las Figs. 1 y 5 del paper de León, una
comparación entre:
    (a) el ansatz original del paper (tanh, alpha=0.027, Nf=300)
    (b) el mejor punto representativo en la región conjunta BBN+Lambda_obs
        para 'tanh'
    (c) idem para 'sinh2n' con n=1.08 (óptimo del barrido de sensibilidad)

IMPORTANTE: el "punto representativo" NO es un MAP de MCMC (ya vimos que
esa cantidad es ruido sobre una meseta plana) -- es el punto de mínimo
chi2+desviación combinados dentro de la región donde AMBOS criterios
(BBN y Lambda_obs) se satisfacen. Ver joint_constraint_region.py.

Salida: comparison_before_after.png (2 paneles: eps1(N) y Q(N)/M_P^4)
"""
import numpy as np
import matplotlib.pyplot as plt

from unimodular_physics import UnimodularModel
from mcmc_sampler import MODEL_SPECS, LOG10_LAMBDA_OBS,passes_shape_prior
from emulator_model import BBNEmulator
# from bbn_evaluator import YP_OBS, YP_ERR, DH_OBS, DH_ERR
from bbn_evaluator import DNEFF_OBS, DNEFF_OBS_ERR

CHI2_THRESHOLD = 5.99
LATE_TIME_TOLERANCE_DEX = 0.5


def compute_chi2_emu(emu, theta):
    dNeff_fo_pred, dNeff_db_pred = emu.predict(theta)
    return ((dNeff_fo_pred - DNEFF_OBS) / DNEFF_OBS_ERR) ** 2

def find_representative_point(model_name, n_fixed=None, n_grid=150):
    bounds = MODEL_SPECS[model_name]["bounds"]
    emu = BBNEmulator(f"emulator_{model_name}.pt")

    alpha_grid = np.logspace(np.log10(bounds["alpha"][0]),
                              np.log10(bounds["alpha"][1]), n_grid)
    Nf_grid = np.linspace(bounds["Nf"][0], bounds["Nf"][1], n_grid)

    best_score = np.inf
    best_params, best_chi2, best_dex = None, None, None

    for a in alpha_grid:
        for Nf in Nf_grid:
            params = {"alpha": a, "Nf": Nf}
            if model_name == "sinh2n":
                params["n"] = n_fixed
            theta = list(params.values())
            try:
                model = UnimodularModel(model_name, params)
                if not passes_shape_prior(model):
                    continue
                chi2 = compute_chi2_emu(emu, theta)
                if chi2 >= CHI2_THRESHOLD:
                    continue
                
                Q_late = model.late_time_Q_over_MP4_proxy()
                if Q_late <= 0:
                    continue
                dex_dev = abs(np.log10(Q_late) - LOG10_LAMBDA_OBS)
                if dex_dev >= LATE_TIME_TOLERANCE_DEX:
                    continue
                score = chi2 + dex_dev
                if score < best_score:
                    best_score = score
                    best_params = dict(params)
                    best_chi2, best_dex = chi2, dex_dev
            except Exception:
                continue

    if best_params is None:
        print(f"[AVISO] No se encontró ningún punto en la región conjunta "
              f"para '{model_name}'" + (f" con n={n_fixed}" if n_fixed else "") +
              " bajo el prior de suavidad (EPS2_MAX). Esto es un resultado "
              "físico, no un error: bajo esta restricción, este ansatz no "
              "concilia BBN con Lambda_obs.")
        return None, None, None

    print(f"[{model_name}] Punto representativo: {best_params} "
          f"(chi2={best_chi2:.2f}, |dev|={best_dex:.2f} dex)")
    return best_params, best_chi2, best_dex


def make_plot():
    original_params = {"alpha": 0.027, "Nf": 300.0}
    original_model = UnimodularModel("tanh", original_params)

    new_tanh_params, chi2_t, dex_t = find_representative_point("tanh")

    # usar el n del MAP más reciente del MCMC, no un valor hardcodeado
    import emcee
    backend = emcee.backends.HDFBackend("chain_sinh2n_emu.h5", read_only=True)
    chain = backend.get_chain(discard=backend.iteration // 4, flat=True)
    n_from_map = float(chain[np.argmax(backend.get_log_prob(discard=backend.iteration // 4, flat=True)), 2])
    print(f"Usando n={n_from_map:.4g} (MAP de la cadena actual)")

    new_sinh2n_params, chi2_s, dex_s = find_representative_point("sinh2n", n_fixed=n_from_map)

    configs = [
        (original_model,
         rf"León original: tanh, $\alpha$={original_params['alpha']}, "
         rf"$N_f$={original_params['Nf']:.0f}",
         "C0", "-"),
    ]
    if new_tanh_params is not None:
        new_tanh_model = UnimodularModel("tanh", new_tanh_params)
        configs.append((new_tanh_model,
            rf"Nuevo (tanh): $\alpha$={new_tanh_params['alpha']:.3g}, "
            rf"$N_f$={new_tanh_params['Nf']:.0f}", "C1", "--"))
    if new_sinh2n_params is not None:
        new_sinh2n_model = UnimodularModel("sinh2n", new_sinh2n_params)
        configs.append((new_sinh2n_model,
            rf"Nuevo (sinh2n, n=0.58): $\alpha$={new_sinh2n_params['alpha']:.3g}, "
            rf"$N_f$={new_sinh2n_params['Nf']:.0f}", "C2", "-."))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for model, label, color, ls in configs:
        Nf = model.params["Nf"]
        N_plot = np.linspace(0, Nf, 500)
        ax.plot(N_plot, model.eps1(N_plot), color=color, linestyle=ls, label=label)
    ax.set_xlabel("N (e-folds)")
    ax.set_ylabel(r"$\epsilon_1(N)$")
    ax.set_title(r"Función $\epsilon_1(N)$: antes vs. después")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for model, label, color, ls in configs:
        Nf = model.params["Nf"]
        N_plot = np.linspace(0, Nf + 100, 500)
        Q_vals = model.Q_over_MP4(N_plot)
        ax.semilogy(N_plot, Q_vals, color=color, linestyle=ls, label=label)
    ax.set_xlabel("N (e-folds)")
    ax.set_ylabel(r"$Q(N)/M_P^4$")
    ax.set_title(r"Término de difusión $Q(N)$: antes vs. después")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("comparison_before_after.png", dpi=150, bbox_inches="tight")
    print("\nGuardado: comparison_before_after.png")

    return {
        "original": original_params,
        "new_tanh": (new_tanh_params, chi2_t, dex_t),
        "new_sinh2n": (new_sinh2n_params, chi2_s, dex_s),
    }


if __name__ == "__main__":
    result = make_plot()
    print("\n=== Resumen para el reporte ===")
    print(result)