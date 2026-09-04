"""
mcmc_sampler.py
===============
MCMC con emcee para AMBAS familias de eps1(N):
    - "tanh"   : parámetros libres (alpha, Nf)         -> 2D
    - "sinh2n" : parámetros libres (alpha, Nf, n)       -> 3D

Likelihood: chi2 de BBN (Yp, D/H) de bbn_evaluator.bbn_chi2().

Priors:
    - Planos dentro de rangos físicamente razonables.
    - CORTE DURO (log_prior = -inf) si:
        * Nf < 60           (no resuelve horizonte/planitud, ref. [75] del paper)
        * min(eps1) durante [0,Nf] no baja de 0.05 (no hay inflación real)
        * eps1 sale de (0,2) en algún punto del dominio (rho o Q negativos)
    - SUAVE (gaussiano en log10): liga Q(N0_proxy) al orden de magnitud de
      la constante cosmológica observada hoy, con sigma generoso (2 dex)
      para no sobre-restringir -- es un "guardarail", no una medición.

      APROXIMACIÓN EXPLÍCITA: usamos N0_proxy = Nf + 62.3 (calibrado sobre
      el resultado del paper) en vez de resolver el matching completo
      materia-radiación (Sec. V) para cada muestra. Es más rápido pero
      menos preciso que el procedimiento completo del paper; queda como
      posible refinamiento (Fase 2b) si hace falta.

ADVERTENCIA DE PERFORMANCE: cada log_likelihood construye una spline y
hace 2 root-findings con brentq -> unos pocos ms por evaluación. Con
nwalkers~32 y nsteps~3000-5000 esto puede tardar minutos-horas en CPU.
Esta es precisamente la razón de ser de la Fase 3 (emulador NN): una vez
entrenado, reemplaza esta función por una inferencia de red en <1ms.
"""
import os
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from unimodular_physics import UnimodularModel
from bbn_evaluator import bbn_chi2

# ----------------------------------------------------------------------
# Constante cosmológica observada hoy, en unidades M_P^2 (calculada, no
# hardcodeada al "~1e-122" redondeado del paper)
# ----------------------------------------------------------------------
H0_OVER_MP = 5.9776e-61   # H0 = 67.3 km/s/Mpc, en unidades de M_P
OMEGA_LAMBDA_0 = 0.6889
LAMBDA_OBS_OVER_MP2 = 3.0 * H0_OVER_MP**2 * OMEGA_LAMBDA_0
LOG10_LAMBDA_OBS = np.log10(LAMBDA_OBS_OVER_MP2)
SIGMA_LATE_DEX = 2.0   # ancho del prior suave, en órdenes de magnitud

# ----------------------------------------------------------------------
# Especificación de modelos y priors (rangos planos)
# ----------------------------------------------------------------------
MODEL_SPECS = {
    "tanh": {
        "ndim": 2,
        "labels": [r"$\alpha$", r"$N_f$"],
        "bounds": {"alpha": (1e-4, 1.0), "Nf": (60.0, 400.0)},
        "log_param": {"alpha": True, "Nf": False},   # log-uniforme en alpha
    },
    "sinh2n": {
        "ndim": 3,
        "labels": [r"$\alpha$", r"$N_f$", r"$n$"],
        "bounds": {"alpha": (1e-4, 1.0), "Nf": (60.0, 400.0), "n": (0.5, 6.0)},
        "log_param": {"alpha": True, "Nf": False, "n": False},
    },
}


def theta_to_params(theta, model_name):
    spec = MODEL_SPECS[model_name]
    names = list(spec["bounds"].keys())
    return dict(zip(names, theta))


# ----------------------------------------------------------------------
# Prior
# ----------------------------------------------------------------------
def log_prior(theta, model_name):
    spec = MODEL_SPECS[model_name]
    params = theta_to_params(theta, model_name)

    # 1. límites planos
    for name, val in params.items():
        lo, hi = spec["bounds"][name]
        if not (lo <= val <= hi):
            return -np.inf

    # 2. construir el modelo (barato: solo eps1 en grilla, sin root-finding aún)
    try:
        model = UnimodularModel(model_name, params)
    except Exception:
        return -np.inf

    if not model.is_valid:
        return -np.inf  # eps1 salió de (0,2) en algún punto

    # 3. debe haber inflación real
    if model.min_eps1_during_inflation() > 0.05:
        return -np.inf
    
    # 3.5. chequeo barato: ¿este modelo llega a T ~ escala de BBN
    # dentro del dominio construido (N_max)? Si a N_max-1 la T todavía
    # es mucho mayor que la ventana de BBN, rechazamos por prior en vez
    # de esperar que N_of_T tire ValueError más adelante.
    try:
        T_at_Nmax = model.T_of_N(model.N_max - 1.0)
    except Exception:
        return -np.inf
    if T_at_Nmax > 1e-2:   # MeV; el techo superior de la ventana de BBN es 10 MeV,
                           # pero pedimos que YA haya bajado del piso (0.01 MeV)
                           # para asegurar margen sobre todo el rango [0.01,10] MeV
        return -np.inf

    # 4. prior suave de tiempos tardíos: Q(N0_proxy) cerca de Lambda_obs
    try:
        Q_late = model.late_time_Q_over_MP4_proxy()
        if Q_late <= 0 or not np.isfinite(Q_late):
            return -np.inf
        log10_Q_late = np.log10(Q_late)
    except Exception:
        return -np.inf

    lp_late = -0.5 * ((log10_Q_late - LOG10_LAMBDA_OBS) / SIGMA_LATE_DEX) ** 2

    # jacobiano por el log-prior en alpha (log-uniforme)
    lp_jac = 0.0
    for name, is_log in spec["log_param"].items():
        if is_log:
            lp_jac += -np.log(params[name])

    return lp_late + lp_jac


# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------
def log_likelihood(theta, model_name):
    params = theta_to_params(theta, model_name)
    try:
        model = UnimodularModel(model_name, params)
        if not model.is_valid:
            return -np.inf
        result = bbn_chi2(model)
        chi2 = result["chi2_total"]
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2
    except Exception:
        return -np.inf


def log_posterior(theta, model_name):
    lp = log_prior(theta, model_name)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, model_name)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ----------------------------------------------------------------------
# Posiciones iniciales (informadas por la estimación analítica
# Gamma(T) ~ exp[-(4/3) alpha * DeltaN], DeltaN ~ 38-45, para no arrancar
# en una región de prior probability ~0)
# ----------------------------------------------------------------------
def initial_walkers(model_name, nwalkers, seed=0):
    rng = np.random.default_rng(seed)
    if model_name == "tanh":
        center = np.array([0.10, 280.0])       # alpha mayor que el paper (0.027)
        spread = np.array([0.05, 40.0])
    elif model_name == "sinh2n":
        center = np.array([0.10, 280.0, 2.0])
        spread = np.array([0.05, 40.0, 1.0])
    else:
        raise ValueError(model_name)

    ndim = MODEL_SPECS[model_name]["ndim"]
    pos = center + spread * rng.standard_normal((nwalkers, ndim))

    # clip a los bounds para evitar -inf en la primera evaluación
    bounds = MODEL_SPECS[model_name]["bounds"]
    for i, name in enumerate(bounds.keys()):
        lo, hi = bounds[name]
        pos[:, i] = np.clip(pos[:, i], lo * 1.01, hi * 0.99)
    return pos


# ----------------------------------------------------------------------
# Runner principal
# ----------------------------------------------------------------------
def run_mcmc(model_name, nwalkers=32, nsteps=4000, seed=0, resume=True):
    ndim = MODEL_SPECS[model_name]["ndim"]
    backend_file = f"chain_{model_name}.h5"
    file_exists = os.path.exists(backend_file)

    backend = emcee.backends.HDFBackend(backend_file)

    if resume and file_exists and backend.iteration > 0:
        print(f"Reanudando cadena existente en '{backend_file}' "
              f"({backend.iteration} pasos ya guardados)...")
        pos0 = None
    else:
        backend.reset(nwalkers, ndim)
        pos0 = initial_walkers(model_name, nwalkers, seed)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(model_name,), backend=backend
    )

    print(f"Corriendo MCMC para modelo '{model_name}' "
          f"({nwalkers} walkers x {nsteps} steps adicionales)...")
    sampler.run_mcmc(pos0, nsteps, progress=True)

    # --- convergencia: tiempo de autocorrelación integrado ---
    try:
        tau = sampler.get_autocorr_time(tol=0)
        print(f"Tiempo de autocorrelación integrado (por parámetro): {tau}")
        burnin = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
    except emcee.autocorr.AutocorrError as e:
        print(f"[AVISO] No se pudo estimar tau de forma confiable: {e}")
        print("        La cadena probablemente necesita más pasos. "
              "Usando burn-in/thin conservadores por defecto.")
        burnin = nsteps // 4
        thin = 10

    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"Fracción de aceptación promedio: {acc_frac:.3f} "
          f"(ideal: ~0.2-0.5)")

    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    print(f"Muestras post burn-in/thin: {samples.shape[0]}")

    return sampler, samples


def make_corner_plot(samples, model_name, filename=None):
    labels = MODEL_SPECS[model_name]["labels"]
    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt=".3g",
        # corner por defecto dibuja niveles equivalentes a 1,2,3 sigma en 2D
        # (fracciones 0.393, 0.865, 0.988 -> aprox. 1,2,3 sigma-gaussianos)
    )
    fig.suptitle(f"Posterior BBN — modelo '{model_name}'", y=1.02)
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Corner plot guardado en: {filename}")
    plt.show()


if __name__ == "__main__":
    for model_name in ["tanh", "sinh2n"]:
        print(f"\n{'='*60}\nMODELO: {model_name}\n{'='*60}")
        sampler, samples = run_mcmc(model_name, nwalkers=32, nsteps=5000)
        make_corner_plot(samples, model_name, filename=f"corner_{model_name}.png")

        best_idx = np.argmax([log_posterior(s, model_name) for s in samples[:200]])
        print(f"Ejemplo de punto de alta probabilidad: {samples[best_idx]}")