"""
compare_chains.py
==================
Compara cuantiles del posterior entre la cadena de física exacta y la
cadena acelerada con el emulador, leyendo directamente los backends HDF5
(no hace falta volver a correr nada).

USO:
    python compare_chains.py --model tanh
    python compare_chains.py --model sinh2n
"""

import argparse
import numpy as np
import emcee

from mcmc_sampler import MODEL_SPECS


def load_chain(filename):
    backend = emcee.backends.HDFBackend(filename, read_only=True)
    n_iter = backend.iteration
    if n_iter == 0:
        return None, 0

    try:
        tau = backend.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
    except Exception:
        tau = None
        burnin = n_iter // 4
        thin = 1

    samples = backend.get_chain(discard=burnin, thin=thin, flat=True)
    return samples, n_iter, tau


def compare(model_name):
    labels = MODEL_SPECS[model_name]["labels"]

    exact_file = f"chain_{model_name}.h5"
    emu_file = f"chain_{model_name}_emu.h5"

    exact_out = load_chain(exact_file)
    emu_out = load_chain(emu_file)

    print(f"\n=== Comparación de posteriors: '{model_name}' ===\n")

    if exact_out[1] == 0:
        print(f"[AVISO] '{exact_file}' no tiene pasos guardados (0 iteraciones). "
              "No se puede comparar todavía -- correr/terminar el MCMC exacto primero.")
        return

    samples_exact, n_iter_exact, tau_exact = exact_out
    samples_emu, n_iter_emu, tau_emu = emu_out

    print(f"Física exacta : {n_iter_exact} pasos guardados, "
          f"tau_max={np.max(tau_exact) if tau_exact is not None else 'N/A'}, "
          f"{len(samples_exact)} muestras post burn-in/thin")
    print(f"Emulador      : {n_iter_emu} pasos guardados, "
          f"tau_max={np.max(tau_emu) if tau_emu is not None else 'N/A'}, "
          f"{len(samples_emu)} muestras post burn-in/thin")

    print(f"\n{'Parametro':<12}{'Exacta (16/50/84%)':<35}{'Emulador (16/50/84%)':<35}")
    for i, label in enumerate(labels):
        q_exact = np.percentile(samples_exact[:, i], [16, 50, 84])
        q_emu = np.percentile(samples_emu[:, i], [16, 50, 84])
        str_exact = f"{q_exact[0]:.4g} / {q_exact[1]:.4g} / {q_exact[2]:.4g}"
        str_emu = f"{q_emu[0]:.4g} / {q_emu[1]:.4g} / {q_emu[2]:.4g}"
        print(f"{label:<12}{str_exact:<35}{str_emu:<35}")

        # ¿la mediana del emulador cae dentro del intervalo 16-84% de la exacta?
        median_emu = q_emu[1]
        within = q_exact[0] <= median_emu <= q_exact[2]
        flag = "OK" if within else "*** DESPLAZADO ***"
        print(f"{'':<12}mediana emulador {'dentro' if within else 'FUERA'} "
              f"del intervalo 1-sigma de la exacta -> {flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    args = parser.parse_args()
    compare(args.model)