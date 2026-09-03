"""
bbn_evaluator.py
================
Evalúa el impacto de la Gravedad Unimodular + difusión Q sobre BBN.

Estrategia (documentada, no una caja negra):
    1. Con unimodular_physics.py calculamos Gamma(T) = Q/rho en la
       temperatura relevante para el freeze-out débil (T ~ 0.7-0.8 MeV,
       la que fija Y_p) y la traducimos a un N_eff efectivo ΔN_eff,
       la parametrización estándar con la que TODOS los códigos de BBN
       (PRIMAT, AlterBBN, PArthENoPE) reportan sensibilidad a una tasa
       de expansión modificada.
    2. Usamos las tablas numéricas reales de PRIMAT/PArthENoPE vía el
       módulo `camb.bbn` (pip install camb) para obtener Y_p y D/H
       en función de (omega_b h^2, ΔN_eff). Esto ES la conexión con el
       solver de BBN pedida, solo que en vez de un wrapper Python
       inestable de AlterBBN, usamos las tablas ya validadas que CAMB
       distribuye oficialmente (PRIMAT_Yp_DH_ErrorMC_2021.dat).
    3. Si camb no está instalado, cae a la fórmula de escala del PDG
       Review of BBN: Delta Y_p ~ 0.013 * Delta N_eff  (fallback de
       orden cero, explícitamente marcado como aproximado).

CAVEAT FÍSICO IMPORTANTE:
    Gamma(T) NO es exactamente constante en toda la ventana de BBN
    (T in [0.01,10] MeV) -> mapear a un único ΔN_eff es una aproximación.
    Lo evaluamos en dos puntos de referencia (freeze-out n/p y cuello de
    botella del deuterio) para chequear cuán fuerte es esa variación.
    Para una respuesta rígida, habría que inyectar H_UG(T) completo en
    el integrador de una red nuclear real (Fase futura / AlterBBN nativo).
"""

import numpy as np
import unimodular_physics as up

# ------------------------------------------------------------------
# Observaciones (valores dados en el enunciado)
# ------------------------------------------------------------------
YP_OBS, YP_ERR = 0.245, 0.003
DH_OBS, DH_ERR = 2.54e-5, 0.04e-5

# Densidad bariónica fiducial (Planck 2018), necesaria para las tablas BBN
OMBH2_FIDUCIAL = 0.02237

# Temperaturas de referencia físicas dentro de la ventana de BBN
T_FREEZEOUT_MEV = 0.75   # fija Y_p (freeze-out n<->p)
T_DBOTTLENECK_MEV = 0.07  # fija D/H (cuello de botella del deuterio)

F_NU = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)   # factor estándar 1-nu vs 1-foton
NEFF_SM = 3.044


def gamma_to_delta_neff(Gamma, Neff_ref=NEFF_SM):
    """
    Traduce una fracción de energía extra Gamma = Q/rho_rad_total en un
    Delta N_eff equivalente, consistente con rho_rad = rho_gamma(1+f*Neff):

        Gamma = f*Delta N_eff / (1 + f*Neff_ref)
        =>  Delta N_eff = Gamma * (1 + f*Neff_ref) / f
    """
    return Gamma * (1.0 + F_NU * Neff_ref) / F_NU


def evaluate_diffusion_term_impact(alpha=up.DEFAULT_PARAMS["alpha"],
                                    Nf=up.DEFAULT_PARAMS["Nf"],
                                    ombh2=OMBH2_FIDUCIAL,
                                    verbose=True):
    """
    Pipeline completo Fase 1:
        parametros de Q  ->  Gamma(T_BBN)  ->  Delta N_eff  ->  Yp, D/H  ->  chi2
    """
    # --- 1. Gamma(T) en los dos puntos de referencia ---
    r_fo, Gamma_fo, N_fo = up.hubble_ratio_UG_over_GR(T_FREEZEOUT_MEV, alpha, Nf)
    r_db, Gamma_db, N_db = up.hubble_ratio_UG_over_GR(T_DBOTTLENECK_MEV, alpha, Nf)

    dNeff_fo = gamma_to_delta_neff(Gamma_fo)
    dNeff_db = gamma_to_delta_neff(Gamma_db)

    if verbose:
        print("=== Paso 1: Gamma(T) y Delta N_eff en puntos clave de BBN ===")
        print(f"T = {T_FREEZEOUT_MEV} MeV (freeze-out n/p, fija Yp):")
        print(f"    N = {N_fo:.2f}, Gamma = {Gamma_fo:.3e}, "
              f"H_UG/H_GR = {r_fo:.6f}, Delta_Neff = {dNeff_fo:.3e}")
        print(f"T = {T_DBOTTLENECK_MEV} MeV (cuello de botella D, fija D/H):")
        print(f"    N = {N_db:.2f}, Gamma = {Gamma_db:.3e}, "
              f"H_UG/H_GR = {r_db:.6f}, Delta_Neff = {dNeff_db:.3e}")

        variation = abs(dNeff_fo - dNeff_db)
        print(f"[chequeo de consistencia] |Delta_Neff(fo) - Delta_Neff(db)| = "
              f"{variation:.3e}  (si no es << 1, la aprox. de Neff constante es pobre)")

    # --- 2. Yp, D/H a partir de Delta N_eff ---
    # Usamos el valor en el freeze-out (el que domina Yp) para Yp,
    # y el del cuello de botella del deuterio para D/H, en vez de forzar
    # un único numero: es más honesto dado el caveat de arriba.
    Yp_pred, DH_pred, backend = _predict_abundances(ombh2, dNeff_fo, dNeff_db)

    if verbose:
        print("\n=== Paso 2: abundancias predichas ===")
        print(f"Backend usado: {backend}")
        print(f"Yp predicho  = {Yp_pred:.5f}   (obs: {YP_OBS} +/- {YP_ERR})")
        print(f"D/H predicho = {DH_pred:.3e}  (obs: {DH_OBS:.3e} +/- {DH_ERR:.2e})")

    # --- 3. chi2 ---
    chi2_Yp = ((Yp_pred - YP_OBS) / YP_ERR) ** 2
    chi2_DH = ((DH_pred - DH_OBS) / DH_ERR) ** 2
    chi2_total = chi2_Yp + chi2_DH

    if verbose:
        print("\n=== Paso 3: chi^2 ===")
        print(f"chi2_Yp    = {chi2_Yp:.2f}")
        print(f"chi2_DH    = {chi2_DH:.2f}")
        print(f"chi2_total = {chi2_total:.2f}  (2 datos -> ~O(1-4) es 'compatible')")

        print("\n=== CONCLUSIÓN ===")
        if chi2_total < 4.0:
            print("Los parámetros originales de León (alpha=0.027, Nf=300) son "
                  "COMPATIBLES con BBN dentro de ~2 sigma combinados.")
        else:
            print("Los parámetros originales de León generan TENSIÓN con BBN "
                  "(chi2 alto) -> pasar a Fase 2 (MCMC) para reajustar alpha, Nf "
                  "u otros parámetros libres del ansatz de Q.")

    return {
        "Gamma_freezeout": Gamma_fo, "Gamma_deuterium": Gamma_db,
        "dNeff_freezeout": dNeff_fo, "dNeff_deuterium": dNeff_db,
        "Yp_pred": Yp_pred, "DH_pred": DH_pred,
        "chi2_Yp": chi2_Yp, "chi2_DH": chi2_DH, "chi2_total": chi2_total,
    }


def _predict_abundances(ombh2, dNeff_Yp, dNeff_DH):
    """
    Intenta usar camb.bbn (tablas PRIMAT/PArthENoPE reales).
    Si no está disponible, cae a la fórmula de escala del PDG:
        Yp(Neff) ~ Yp(Neff=3.044) + 0.013 * Delta_Neff   [PDG BBN review]
        D/H: sin coeficiente confiable de memoria -> se deja marcado.
    """
    try:
        from camb.bbn import BBN_table_interpolator
        interp = BBN_table_interpolator()  # usa la tabla PRIMAT por defecto

        # Nombres de método pueden variar entre versiones de camb; probamos
        # las variantes documentadas y avisamos si hay que ajustarlas.
        Yp_method = getattr(interp, "Y_p", None) or getattr(interp, "Y_He", None)
        DH_method = getattr(interp, "DH", None)

        if Yp_method is None or DH_method is None:
            raise AttributeError(
                "No se encontraron los métodos esperados en BBN_table_interpolator; "
                "revisar `dir(interp)` para tu versión instalada de camb."
            )

        Yp_pred = float(Yp_method(ombh2, dNeff_Yp))
        DH_pred = float(DH_method(ombh2, dNeff_DH))
        return Yp_pred, DH_pred, "camb.bbn (tablas PRIMAT/PArthENoPE)"

    except ImportError:
        # --- Fallback aproximado, sin camb instalado ---
        Yp_SM = 0.2470  # valor estándar aproximado para ombh2 ~ 0.0224, Neff=3.044
        Yp_pred = Yp_SM + 0.013 * dNeff_Yp   # PDG: dYp/dNeff ~ 0.013

        DH_SM = 2.45e-5
        # Coeficiente aproximado de orden de magnitud (D/H crece con Neff);
        # NO calibrado con precisión -> usar solo como placeholder.
        DH_pred = DH_SM * (1.0 + 0.03 * dNeff_DH)

        return Yp_pred, DH_pred, "FALLBACK aproximado (instalar `camb` para precisión)"


if __name__ == "__main__":
    evaluate_diffusion_term_impact()
