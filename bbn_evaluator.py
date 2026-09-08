"""
bbn_evaluator.py (v2)
======================
ACTUALIZACIÓN: comparación directa contra Neff observado, en vez de la
cadena Gamma -> Yp,D/H (camb.bbn) -> chi2_Yp+chi2_DH.

Motivo: Goldstein & Hill (2026, arXiv:2603.13226) ya combinan de forma
óptima Yp (LBT Yp Project V, Yeh et al. 2026), D/H, CMB y BAO en un único
número: Neff = 2.990 +- 0.070 (68% CL). Reconstruir esto nosotros mismos
vía camb.bbn introducía error de aproximación adicional e ignoraba la
combinación óptima que ya hicieron los autores. Ahora usamos ese número
directamente como el ÚNICO observable del likelihood.

Se mantiene bbn_chi2_legacy() (método anterior, Yp+D/H vía camb.bbn) solo
para comparación/diagnóstico, con Yp_obs actualizado a Yeh et al. 2026.
"""
import numpy as np

# ------------------------------------------------------------------
# Observable principal (NUEVO): Neff combinado, Goldstein & Hill 2026
# ------------------------------------------------------------------
NEFF_OBS = 2.990
NEFF_OBS_ERR = 0.070
NEFF_SM = 3.044  # valor del Modelo Estándar (sin nueva física)
DNEFF_OBS = NEFF_OBS - NEFF_SM       # = -0.054
DNEFF_OBS_ERR = NEFF_OBS_ERR         # = 0.070

# ------------------------------------------------------------------
# Observables legacy (Yp, D/H por separado) -- solo para diagnóstico
# ------------------------------------------------------------------
YP_OBS, YP_ERR = 0.2458, 0.0013      # Yeh et al. 2026, LBT Yp Project V
DH_OBS, DH_ERR = 2.527e-5, 0.030e-5  # Cooke et al. 2018

OMBH2_FIDUCIAL = 0.02237
T_FREEZEOUT_MEV = 0.75
T_DBOTTLENECK_MEV = 0.07
F_NU = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)


def gamma_to_delta_neff(Gamma, Neff_ref=NEFF_SM):
    return Gamma * (1.0 + F_NU * Neff_ref) / F_NU


def bbn_chi2(model):
    """
    NUEVO criterio principal: chi2 = ((dNeff_modelo - DNEFF_OBS)/DNEFF_OBS_ERR)^2,
    evaluado con dNeff_fo (congelamiento débil n<->p) como proxy principal
    -- es el canal dominante en la sensibilidad de Neff a nueva física,
    igual que en el analisis original. dNeff_db se reporta aparte como
    chequeo de consistencia (variación del proxy de 2 puntos), NO se suma
    al chi2 para evitar doble conteo de la misma physica.
    """
    r_fo, Gamma_fo, N_fo = model.hubble_ratio(T_FREEZEOUT_MEV)
    r_db, Gamma_db, N_db = model.hubble_ratio(T_DBOTTLENECK_MEV)

    dNeff_fo = gamma_to_delta_neff(Gamma_fo)
    dNeff_db = gamma_to_delta_neff(Gamma_db)

    chi2 = ((dNeff_fo - DNEFF_OBS) / DNEFF_OBS_ERR) ** 2

    return {
        "Gamma_fo": Gamma_fo, "Gamma_db": Gamma_db,
        "dNeff_fo": dNeff_fo, "dNeff_db": dNeff_db,
        "chi2_total": chi2,
        "backend": "Neff directo (Goldstein & Hill 2026)",
    }


def bbn_chi2_legacy(model, ombh2=OMBH2_FIDUCIAL):
    """Método anterior (Yp+D/H vía camb.bbn), solo para comparación."""
    r_fo, Gamma_fo, N_fo = model.hubble_ratio(T_FREEZEOUT_MEV)
    r_db, Gamma_db, N_db = model.hubble_ratio(T_DBOTTLENECK_MEV)
    dNeff_fo = gamma_to_delta_neff(Gamma_fo)
    dNeff_db = gamma_to_delta_neff(Gamma_db)

    try:
        from camb.bbn import BBN_table_interpolator
        interp = BBN_table_interpolator()
        Yp_method = getattr(interp, "Y_p", None) or getattr(interp, "Y_He", None)
        DH_method = getattr(interp, "DH", None)
        Yp_pred = float(Yp_method(ombh2, dNeff_fo))
        DH_pred = float(DH_method(ombh2, dNeff_db))
    except Exception:
        Yp_pred = 0.2470 + 0.013 * dNeff_fo
        DH_pred = 2.45e-5 * (1.0 + 0.03 * dNeff_db)

    chi2_Yp = ((Yp_pred - YP_OBS) / YP_ERR) ** 2
    chi2_DH = ((DH_pred - DH_OBS) / DH_ERR) ** 2
    return {"Yp_pred": Yp_pred, "DH_pred": DH_pred,
            "chi2_Yp": chi2_Yp, "chi2_DH": chi2_DH,
            "chi2_total": chi2_Yp + chi2_DH}


if __name__ == "__main__":
    from unimodular_physics import UnimodularModel, DEFAULT_PARAMS_TANH
    m = UnimodularModel("tanh", DEFAULT_PARAMS_TANH)
    print("=== Nuevo criterio (Neff directo) ===")
    print(bbn_chi2(m))
    print("\n=== Legacy (Yp+D/H via camb.bbn) ===")
    print(bbn_chi2_legacy(m))