"""
unimodular_physics.py
======================
Módulo físico: implementa el sector de fondo (background) del modelo de
inflación por difusión en Gravedad Unimodular de León (arXiv:2202.04029).

Ecuaciones implementadas (numeración del paper):
    eps1(N)                      -> Eq. (34)
    Q(N)/M_P^4, rho(N)/M_P^4      -> Eqs. (35),(36), reescritas vía Eq.(32)-(33)
                                     en forma log-estable (ver notas abajo)
    Gamma(N) = Q/rho = 2/eps1 - 1 -> Eq. (18) invertida

Todo se calcula en unidades de Planck reducido M_P, y se convierte a MeV
cuando hace falta acoplar con la física de BBN.

NOTA NUMÉRICA:
    Para N ~ 300-370 (rango relevante para BBN/post-inflación) los factores
    exp[...] y cosh(...)^(3/alpha) individualmente sub/sobre-flotan en
    float64 (Q/rho ~ 1e-120). Por eso todo el cálculo de rho(N) y Q(N) se
    hace en log-espacio y solo se exponencía al final.
"""

import numpy as np
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# Constantes físicas y de conversión
# ----------------------------------------------------------------------
M_P_MEV = 2.435e21          # Masa de Planck reducida, en MeV (M_P = 2.435e18 GeV)
PI2_OVER_30 = np.pi**2 / 30.0

# Parámetros del ansatz de León et al. (Sec. III-V del paper),
# calibrados para reproducir el orden de magnitud de la CC observada hoy.
DEFAULT_PARAMS = {
    "alpha": 0.027,   # parámetro libre alpha
    "Nf": 300.0,      # e-folds totales de inflación
    "N0": 362.3,      # e-folds hoy (a0 = e^N0 * a_ini)
    "Neq": 357.3,     # e-folds en igualdad materia-radiación (T_eq ~ 0.5 eV)
}


# ----------------------------------------------------------------------
# Utilidades numéricas
# ----------------------------------------------------------------------
def _safe_logcosh(x):
    """log(cosh(x)) estable para |x| grande."""
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


# ----------------------------------------------------------------------
# Fondo cosmológico: eps1, Q, rho (Eqs. 34, 35, 36)
# ----------------------------------------------------------------------
def eps1(N, alpha=DEFAULT_PARAMS["alpha"], Nf=DEFAULT_PARAMS["Nf"]):
    """Primer Hubble-flow-function, Eq. (34)."""
    return 1.0 + np.tanh((2.0 / 3.0) * alpha * (N - Nf)) + np.exp(-4.0 * alpha * N)


def _log_rho_and_Q(N, alpha, Nf):
    """
    Devuelve (log(rho/M_P^4), log(Q/M_P^4)) usando la reescritura
    log-estable de las Eqs. (32),(33):

        rho/M_P^4 = eps1(N) * exp(E(N)) * R(N)^(3/alpha)
        Q/M_P^4   = [2-eps1(N)] * exp(E(N)) * R(N)^(3/alpha)

    con E(N) = (1/2alpha)(-1 + exp(-4 alpha N) - 4 alpha N)
    y   R(N) = cosh(2 alpha Nf/3) / cosh(2 alpha (N-Nf)/3)
    """
    e1 = eps1(N, alpha, Nf)
    E = (1.0 / (2.0 * alpha)) * (-1.0 + np.exp(-4.0 * alpha * N) - 4.0 * alpha * N)
    logR = _safe_logcosh(2.0 * alpha * Nf / 3.0) - _safe_logcosh(2.0 * alpha * (N - Nf) / 3.0)
    common = E + (3.0 / alpha) * logR

    log_rho = np.log(e1) + common
    log_Q = np.log(2.0 - e1) + common
    return log_rho, log_Q


def rho_over_MP4(N, alpha=DEFAULT_PARAMS["alpha"], Nf=DEFAULT_PARAMS["Nf"]):
    """Densidad de energía de radiación rho(N), en unidades de M_P^4."""
    log_rho, _ = _log_rho_and_Q(N, alpha, Nf)
    return np.exp(log_rho)


def Q_over_MP4(N, alpha=DEFAULT_PARAMS["alpha"], Nf=DEFAULT_PARAMS["Nf"]):
    """Término de difusión Q(N), en unidades de M_P^4."""
    _, log_Q = _log_rho_and_Q(N, alpha, Nf)
    return np.exp(log_Q)


def Gamma_of_N(N, alpha=DEFAULT_PARAMS["alpha"], Nf=DEFAULT_PARAMS["Nf"]):
    """
    Gamma(N) = Q/rho, calculado de la forma robusta (Eq. 18 invertida):
        Gamma = 2/eps1(N) - 1
    Evita el cociente de dos números extremadamente pequeños.
    """
    e1 = eps1(N, alpha, Nf)
    return 2.0 / e1 - 1.0


# ----------------------------------------------------------------------
# Grados de libertad relativistas g*(T) (aproximación de dos mesetas)
# ----------------------------------------------------------------------
def g_star(T_MeV, g_high=10.75, g_low=3.36, T_transition=0.35, width=0.15):
    """
    Aproximación suave (tanh) de g*(T) para el rango de BBN.
        g_high = 10.75  -> fotones + e+/e- + 3 nu (T >~ 1-2 MeV)
        g_low  = 3.36   -> fotones + 3 nu ya desacoplados y fríos (T <~ 0.05 MeV)
    ADVERTENCIA: esto es una aproximación ilustrativa de orden cero.
    Para un cálculo de precisión reemplazar por una tabla real de g*(T).
    """
    x = (np.log(T_MeV) - np.log(T_transition)) / width
    frac = 0.5 * (1.0 + np.tanh(x))
    return g_low + (g_high - g_low) * frac


# ----------------------------------------------------------------------
# Mapeo N <-> T (temperatura de fotones/plasma, en MeV)
# ----------------------------------------------------------------------
def _rho_rad_MeV4_of_N(N, alpha, Nf):
    """rho_rad(N) convertido a MeV^4."""
    return rho_over_MP4(N, alpha, Nf) * M_P_MEV**4


def T_of_N(N, alpha=DEFAULT_PARAMS["alpha"], Nf=DEFAULT_PARAMS["Nf"]):
    """
    Dado N, resuelve T (en MeV) de:
        rho_rad(N) = (pi^2/30) g*(T) T^4
    """
    rho_target = _rho_rad_MeV4_of_N(N, alpha, Nf)

    def f(logT):
        T = np.exp(logT)
        return PI2_OVER_30 * g_star(T) * T**4 - rho_target

    # Bracket amplio en log(T); T esperado entre ~1e-6 y ~1e4 MeV para N razonables
    logT_sol = brentq(f, np.log(1e-8), np.log(1e6))
    return np.exp(logT_sol)


def N_of_T(T_MeV, alpha=DEFAULT_PARAMS["alpha"], Nf=DEFAULT_PARAMS["Nf"],
           N_bracket=(DEFAULT_PARAMS["Nf"], DEFAULT_PARAMS["Neq"] + 5.0)):
    """
    Inversa de T_of_N: dado T (MeV) en el rango de radiación (post-inflación,
    antes de igualdad materia-radiación), encuentra N.
    N_bracket debe encerrar la raíz (por defecto: entre el fin de inflación
    y un poco después de igualdad).
    """
    def f(N):
        return T_of_N(N, alpha, Nf) - T_MeV

    return brentq(f, N_bracket[0], N_bracket[1])


# ----------------------------------------------------------------------
# Factor de modificación de la tasa de expansión H_UG/H_GR
# ----------------------------------------------------------------------
def hubble_ratio_UG_over_GR(T_MeV, alpha=DEFAULT_PARAMS["alpha"],
                             Nf=DEFAULT_PARAMS["Nf"],
                             N_bracket=(DEFAULT_PARAMS["Nf"], DEFAULT_PARAMS["Neq"] + 5.0)):
    """
    r(T) = H_UG(T) / H_GR(T) = sqrt(1 + Gamma(T))

    A partir de 3 H_UG^2 M_P^2 = rho + Q  y  3 H_GR^2 M_P^2 = rho
    (Lambda_* = 0, ver Eq. 9 del paper).
    """
    N = N_of_T(T_MeV, alpha, Nf, N_bracket)
    Gamma = Gamma_of_N(N, alpha, Nf)
    return np.sqrt(1.0 + Gamma), Gamma, N


# ----------------------------------------------------------------------
# Autotest / validación contra los valores reportados en el paper (Sec. V)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = DEFAULT_PARAMS
    print("=== Validación contra Sec. V del paper (alpha=0.027, Nf=300) ===")

    # El paper reporta Q(N0) ~ 1e-122 M_P^4 para N0 = 362.3
    Q_N0 = Q_over_MP4(p["N0"], p["alpha"], p["Nf"])
    print(f"Q(N0={p['N0']})/M_P^4 = {Q_N0:.3e}   (paper: ~1e-122)")

    rho_N0 = rho_over_MP4(p["N0"], p["alpha"], p["Nf"])
    print(f"rho_rad(N0)/M_P^4    = {rho_N0:.3e}   (paper: ~1e-124)")

    # Chequeo de Gamma(N0) ~ Q/rho ~ 100 (consistente con Q dominando hoy)
    print(f"Gamma(N0) = Q/rho    = {Q_N0/rho_N0:.3e}")

    # T(Neq) debería dar ~0.5 eV = 5e-4 MeV
    T_eq = T_of_N(p["Neq"], p["alpha"], p["Nf"])
    print(f"T(Neq={p['Neq']}) = {T_eq*1e6:.2f} eV  (target: ~0.5e6 eV = 0.5 MeV... "
          f"en realidad 0.5 eV, ver nota abajo)")
