"""
bbn_evaluator.py
================
Ahora recibe un objeto UnimodularModel (de unimodular_physics.py) en vez de
(alpha, Nf) hardcodeados, para poder usarse con cualquier familia eps1(N).
"""

import numpy as np

YP_OBS, YP_ERR = 0.245, 0.003
DH_OBS, DH_ERR = 2.54e-5, 0.04e-5
OMBH2_FIDUCIAL = 0.02237

T_FREEZEOUT_MEV = 0.75
T_DBOTTLENECK_MEV = 0.07
_CACHED_INTERPOLATOR = None

F_NU = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
NEFF_SM = 3.044


def gamma_to_delta_neff(Gamma, Neff_ref=NEFF_SM):
    return Gamma * (1.0 + F_NU * Neff_ref) / F_NU


def _predict_abundances(ombh2, dNeff_Yp, dNeff_DH):
    try:
        from camb.bbn import BBN_table_interpolator
        interp = BBN_table_interpolator()
        Yp_method = getattr(interp, "Y_p", None) or getattr(interp, "Y_He", None)
        DH_method = getattr(interp, "DH", None)
        if Yp_method is None or DH_method is None:
            raise AttributeError("Métodos esperados no encontrados en camb.bbn")
        return (float(Yp_method(ombh2, dNeff_Yp)),
                float(DH_method(ombh2, dNeff_DH)),
                "camb.bbn")
    except ImportError:
        Yp_pred = 0.2470 + 0.013 * dNeff_Yp
        DH_pred = 2.45e-5 * (1.0 + 0.03 * dNeff_DH)
        return Yp_pred, DH_pred, "FALLBACK aproximado"


def bbn_chi2(model, ombh2=OMBH2_FIDUCIAL):
    """
    Dado un UnimodularModel ya construido, devuelve el diccionario con
    Gamma, Delta_Neff, Yp, D/H y chi2. Lanza excepción si el modelo no
    es físicamente evaluable en la ventana de BBN (se captura en el MCMC).
    """
    r_fo, Gamma_fo, N_fo = model.hubble_ratio(T_FREEZEOUT_MEV)
    r_db, Gamma_db, N_db = model.hubble_ratio(T_DBOTTLENECK_MEV)

    dNeff_fo = gamma_to_delta_neff(Gamma_fo)
    dNeff_db = gamma_to_delta_neff(Gamma_db)

    Yp_pred, DH_pred, backend = _predict_abundances(ombh2, dNeff_fo, dNeff_db)

    chi2_Yp = ((Yp_pred - YP_OBS) / YP_ERR) ** 2
    chi2_DH = ((DH_pred - DH_OBS) / DH_ERR) ** 2

    return {
        "Gamma_fo": Gamma_fo, "Gamma_db": Gamma_db,
        "dNeff_fo": dNeff_fo, "dNeff_db": dNeff_db,
        "Yp_pred": Yp_pred, "DH_pred": DH_pred,
        "chi2_Yp": chi2_Yp, "chi2_DH": chi2_DH,
        "chi2_total": chi2_Yp + chi2_DH,
        "backend": backend,
    }

def _get_camb_interpolator():
    global _CACHED_INTERPOLATOR
    if _CACHED_INTERPOLATOR is None:
        from camb.bbn import BBN_table_interpolator
        _CACHED_INTERPOLATOR = BBN_table_interpolator()
    return _CACHED_INTERPOLATOR


def _predict_abundances(ombh2, dNeff_Yp, dNeff_DH):
    try:
        interp = _get_camb_interpolator()
        Yp_method = getattr(interp, "Y_p", None) or getattr(interp, "Y_He", None)
        DH_method = getattr(interp, "DH", None)
        if Yp_method is None or DH_method is None:
            raise AttributeError("Métodos esperados no encontrados en camb.bbn")
        return (float(Yp_method(ombh2, dNeff_Yp)),
                float(DH_method(ombh2, dNeff_DH)),
                "camb.bbn")
    except ImportError:
        Yp_pred = 0.2470 + 0.013 * dNeff_Yp
        DH_pred = 2.45e-5 * (1.0 + 0.03 * dNeff_DH)
        return Yp_pred, DH_pred, "FALLBACK aproximado"

if __name__ == "__main__":
    from unimodular_physics import UnimodularModel, DEFAULT_PARAMS_TANH
    m = UnimodularModel("tanh", DEFAULT_PARAMS_TANH)
    result = bbn_chi2(m)
    for k, v in result.items():
        print(f"{k}: {v}")