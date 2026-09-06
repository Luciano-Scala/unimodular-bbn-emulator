"""
unimodular_physics.py
======================
Motor físico GENÉRICO para el sector de fondo del modelo de inflación por
difusión Q en Gravedad Unimodular (León, arXiv:2202.04029).

Novedad vs. Fase 1: en vez de estar atado al ansatz particular Eq.(34)-(36)
del paper, ahora soporta CUALQUIER función eps1(N; params) que satisfaga
las condiciones físicas de inflación (Sec. III). Esto es posible porque:

    Gamma(N) = Q/rho = 2/eps1(N) - 1                      [Eq. (18)]

es EXACTA para cualquier eps1(N), siempre que w=1/3 (radiación pura) y se
use la ecuación de continuidad Eq.(16). No depende de la forma específica
del ansatz. Lo único que sí requiere el ansatz específico es la NORMALIZACIÓN
absoluta de rho(N) (necesaria para el mapeo N<->T), que ahora se obtiene
integrando numéricamente:

    rho(N)/M_P^4 = eps1(N) * exp[-\\int_0^N 2 eps1(N') dN']              [Eq. (32)]
    Q(N)/M_P^4   = [2-eps1(N)] * exp[-\\int_0^N 2 eps1(N') dN']          [Eq. (33)]

Modelos incluidos:
    "tanh"    -> Eq. (34) del paper, params: alpha, Nf         (2 parámetros)
    "sinh2n"  -> familia generalizada (nota al pie 8), params: alpha, Nf, n
                 eps1(N) = sinh^(2n)[alpha(N-Nf/2)] / sinh^(2n)[alpha*Nf/2]
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

# ----------------------------------------------------------------------
M_P_MEV = 2.435e21
PI2_OVER_30 = np.pi**2 / 30.0

DEFAULT_PARAMS_TANH = {"alpha": 0.027, "Nf": 300.0}

# proxy fijo de "hoy" en e-folds relativos al fin de inflación, calibrado
# sobre el resultado del paper (N0 - Nf = 362.3 - 300 = 62.3).
# Ver aviso de aproximación en mcmc_sampler.py (prior de tiempos tardíos).
N0_MINUS_NF_PROXY = 62.3


# ----------------------------------------------------------------------
# Familia de funciones eps1(N; params)
# ----------------------------------------------------------------------
def eps1_tanh(N, alpha, Nf):
    """Eq. (34) del paper."""
    return 1.0 + np.tanh((2.0 / 3.0) * alpha * (N - Nf)) + np.exp(-4.0 * alpha * N)


def eps1_sinh2n_bump(N, alpha, Nf, n, floor=1e-6):
    """
    La función original de la nota 8, SOLO válida/con sentido en [0, Nf].
    `floor` evita log(0) exacto si el grid pisa justo N=Nf/2 (sinh=0).
    """
    num = np.abs(np.sinh(alpha * (N - Nf / 2.0)))
    den = np.abs(np.sinh(alpha * Nf / 2.0))
    val = (num / den) ** (2.0 * n)
    return np.maximum(val, floor)


def eps1_sinh2n(N, alpha, Nf, n):
    """
    Extensión mía (NO está en el paper) del ansatz de la nota 8, para que
    sea evaluable más allá del fin de inflación:

        N <= Nf : eps1_sinh2n_bump(N)                 [ansatz original]
        N >  Nf : 1 + tanh[(2/3) alpha (N - Nf)]       [misma cola que Eq.34]

    Empalma con continuidad en valor en N=Nf (ambos dan 1) y satura a
    eps1->2 para N>>Nf, dando la misma salida suave hacia radiación que
    el modelo 'tanh'. Es una elección de modelado explícita para que la
    familia sea comparable con 'tanh' en el mismo dominio de BBN.
    """
    N = np.atleast_1d(N).astype(float)
    out = np.empty_like(N)
    pre = N <= Nf
    out[pre] = eps1_sinh2n_bump(N[pre], alpha, Nf, n)
    out[~pre] = 1.0 + np.tanh((2.0 / 3.0) * alpha * (N[~pre] - Nf))
    return out if out.size > 1 else out[0]



EPS1_MODELS = {
    "tanh":   {"func": eps1_tanh,   "param_names": ("alpha", "Nf")},
    "sinh2n": {"func": eps1_sinh2n, "param_names": ("alpha", "Nf", "n")},
}


# ----------------------------------------------------------------------
# g*(T) — igual que Fase 1
# ----------------------------------------------------------------------
def g_star(T_MeV, g_high=10.75, g_low=3.36, T_transition=0.35, width=0.15):
    x = (np.log(T_MeV) - np.log(T_transition)) / width
    frac = 0.5 * (1.0 + np.tanh(x))
    return g_low + (g_high - g_low) * frac


# ----------------------------------------------------------------------
# Clase genérica del modelo
# ----------------------------------------------------------------------
class UnimodularModel:
    """
    Envuelve una elección de (familia eps1, parámetros) y expone:
        eps1(N), Gamma(N), rho_over_MP4(N), Q_over_MP4(N),
        T_of_N(N), N_of_T(T), hubble_ratio(T)

    N_max: dominio de construcción de la spline log(rho). Debe cubrir
           holgadamente [0, Nf] (inflación) y la ventana de BBN
           (Nf, Nf + ~80) sin llegar a rangos donde rho subflote en
           float64 (por eso NO usamos N_max enorme).
    """

    def __init__(self, model_name, params, N_max=None, n_grid=4000):
        if model_name not in EPS1_MODELS:
            raise ValueError(f"Modelo desconocido: {model_name}. "
                              f"Opciones: {list(EPS1_MODELS.keys())}")
        spec = EPS1_MODELS[model_name]
        missing = set(spec["param_names"]) - set(params.keys())
        if missing:
            raise ValueError(f"Faltan parámetros {missing} para el modelo {model_name}")

        self.model_name = model_name
        self.params = dict(params)
        self._eps1_func = spec["func"]

        Nf = params["Nf"]
        self.N_max = N_max if N_max is not None else Nf + N0_MINUS_NF_PROXY + 20.0

        Ngrid = np.linspace(0.0, self.N_max, n_grid)
        eps1_grid = self._eps1_func(Ngrid, **self.params)

        if np.any(eps1_grid <= 0) or np.any(eps1_grid >= 2.0 + 1e-9):
            self._valid = False
            self.eps1_grid = eps1_grid
            self.Ngrid = Ngrid
            return

        self._valid = True
        cum_int = cumulative_trapezoid(2.0 * eps1_grid, Ngrid, initial=0.0)

        # Piso numérico: evita log(0) cuando eps1 satura a 2.0 en float64.
        # 1e-12 es << cualquier Gamma físicamente relevante (Gamma~Q/rho
        # nunca baja tanto en la ventana que nos importa), así que no
        # afecta la física, solo evita el -inf espurio en la spline.
        FLOOR = 1e-12
        log_rho_grid = np.log(np.maximum(eps1_grid, FLOOR)) - cum_int
        log_Q_grid = np.log(np.maximum(2.0 - eps1_grid, FLOOR)) - cum_int

        self._log_rho_spline = CubicSpline(Ngrid, log_rho_grid)
        self._log_Q_spline = CubicSpline(Ngrid, log_Q_grid)
        self.Ngrid = Ngrid
        self.eps1_grid = eps1_grid

    # -- checks de viabilidad física (ver mcmc_sampler.py para el uso) --
    @property
    def is_valid(self):
        return self._valid

    def min_eps1_during_inflation(self):
        """min(eps1) en N in [0, Nf], usando la parte 'bump' original
        (sin el parche de salida que agregué para N>Nf)."""
        if self.model_name == "sinh2n":
            Nf = self.params["Nf"]
            N_check = np.linspace(0, Nf, 2000)
            return np.min(eps1_sinh2n_bump(
                N_check, self.params["alpha"], Nf, self.params["n"]
            ))
        mask = self.Ngrid <= self.params["Nf"]
        return np.min(self.eps1_grid[mask]) if np.any(mask) else np.inf

    # -- física de fondo --
    def eps1(self, N):
        return self._eps1_func(N, **self.params)

    def Gamma_of_N(self, N):
        return 2.0 / self.eps1(N) - 1.0

    def rho_over_MP4(self, N):
        return np.exp(self._log_rho_spline(N))

    def Q_over_MP4(self, N):
        return np.exp(self._log_Q_spline(N))

    # -- mapeo N <-> T --
    def _rho_rad_MeV4_of_N(self, N):
        return self.rho_over_MP4(N) * M_P_MEV**4

    def T_of_N(self, N):
        rho_target = self._rho_rad_MeV4_of_N(N)
        if not np.isfinite(rho_target) or rho_target <= 0:
            raise ValueError(f"rho_rad(N={N}) inválido: {rho_target}")

        def f(logT):
            T = np.exp(logT)
            return PI2_OVER_30 * g_star(T) * T**4 - rho_target

        # Techo ampliado hasta ~2x la escala de Planck (en MeV), porque
        # con eps1 casi nulo durante casi toda la inflación (ansatz
        # sinh2n con floor bajo), rho(N) se mantiene pegada a M_P^4
        # hasta muy cerca de N=Nf -> T puede acercarse a la propia
        # escala de Planck. Piso bajado a 1e-25 MeV (~1e-31 eV) para
        # cubrir sin problema el otro extremo, muy por debajo de BBN.
        logT_lo, logT_hi = np.log(1e-25), np.log(2.0 * M_P_MEV)
        f_lo, f_hi = f(logT_lo), f(logT_hi)

        if f_lo * f_hi > 0:
            raise ValueError(
                f"No se pudo acotar T para N={N}: "
                f"f(1e-25 MeV)={f_lo:.3e}, f({2*M_P_MEV:.2e} MeV)={f_hi:.3e}. "
                "rho_target fuera del rango físico esperado (revisar params)."
            )
        return np.exp(brentq(f, logT_lo, logT_hi))

    def N_of_T(self, T_MeV, N_bracket=None):
        Nf = self.params["Nf"]
        if N_bracket is None:
            N_bracket = (Nf, min(Nf + N0_MINUS_NF_PROXY, self.N_max - 1.0))

        def f(N):
            return self.T_of_N(N) - T_MeV

        f_lo, f_hi = f(N_bracket[0]), f(N_bracket[1])
        if f_lo * f_hi > 0:
            raise ValueError(
                f"T={T_MeV} MeV no bracketed en {N_bracket} "
                f"(f_lo={f_lo:.3e}, f_hi={f_hi:.3e})"
            )
        return brentq(f, N_bracket[0], N_bracket[1])

    def hubble_ratio(self, T_MeV, N_bracket=None):
        """r(T) = H_UG/H_GR = sqrt(1+Gamma(T)), junto con N y Gamma."""
        N = self.N_of_T(T_MeV, N_bracket)
        Gamma = self.Gamma_of_N(N)
        return np.sqrt(1.0 + Gamma), Gamma, N

    def late_time_Q_over_MP4_proxy(self):
        """
        Q(N0)/M_P^4 usando N0 = Nf + N0_MINUS_NF_PROXY como proxy fijo de
        'hoy' (aproximación explícita, ver docstring del módulo y
        mcmc_sampler.py).
        """
        N0_proxy = self.params["Nf"] + N0_MINUS_NF_PROXY
        return self.Q_over_MP4(min(N0_proxy, self.N_max - 1e-6))

    def max_abs_eps2_during_inflation(self, n_check=500):
        """
        eps2 = d(ln eps1)/dN (numérico), evaluado en [0, Nf]. Chequea la
        condición de "transición suave" (Ec. 21 del paper: |eps2|<<1),
        necesaria para que el cálculo del espectro primordial (Sec. IV)
        sea válido. Un eps1(N) tipo escalón (alpha grande) viola esto
        aunque pase el chequeo de min(eps1).
        """
        Nf = self.params["Nf"]
        N_check = np.linspace(1e-3, Nf - 1e-3, n_check)
        eps1_vals = self.eps1(N_check)
        eps1_vals = np.maximum(eps1_vals, 1e-300)  # evita log(0)
        dln_eps1_dN = np.gradient(np.log(eps1_vals), N_check)
        return np.max(np.abs(dln_eps1_dN))


if __name__ == "__main__":
    print("=== Validación: modelo 'tanh' con parámetros del paper ===")
    m = UnimodularModel("tanh", DEFAULT_PARAMS_TANH)
    N0 = DEFAULT_PARAMS_TANH["Nf"] + N0_MINUS_NF_PROXY
    print(f"Q(N0={N0:.1f})/M_P^4 = {m.Q_over_MP4(N0):.3e}  (paper: ~1e-122)")

    print("\n=== Prueba del modelo generalizado 'sinh2n' ===")
    m2 = UnimodularModel("sinh2n", {"alpha": 0.08, "Nf": 300.0, "n": 2.0})
    print(f"min(eps1) en inflación = {m2.min_eps1_during_inflation():.3e}")
    r, Gamma, N = m2.hubble_ratio(0.75)
    print(f"T=0.75 MeV -> N={N:.2f}, Gamma={Gamma:.3e}, H_UG/H_GR={r:.6f}")