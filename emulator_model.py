"""
emulator_model.py
==================
Emulador MLP en PyTorch para reemplazar la física exacta (spline + root
finding + camb.bbn) por una inferencia de red, dentro del MCMC.

Un modelo por familia (tanh, sinh2n). Cada checkpoint guarda junto con los
pesos: las estadísticas de normalización de entrada/salida, para que la
inferencia sea autocontenida (no depende de recalcular stats del dataset).

Entradas  : [log10(alpha), Nf]            (tanh)
            [log10(alpha), Nf, n]         (sinh2n)
Salidas   : [Yp, log10(D/H)]              (log10 en D/H, ver docstring del módulo)

USO:
    python emulator_model.py --model tanh   --csv dataset_tanh.csv
    python emulator_model.py --model sinh2n --csv dataset_sinh2n.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from mcmc_sampler import MODEL_SPECS


# ----------------------------------------------------------------------
# Preprocesamiento: mapeo params físicos <-> features de la red
# ----------------------------------------------------------------------
def params_to_features(df, model_name):
    """
    Convierte las columnas de parámetros físicos del DataFrame a un array
    (N, n_features), aplicando log10 a 'alpha' (ver MODEL_SPECS['log_param']).
    """
    spec = MODEL_SPECS[model_name]
    names = list(spec["bounds"].keys())
    cols = []
    for name in names:
        col = df[name].values.astype(np.float64)
        if spec["log_param"].get(name, False):
            col = np.log10(col)
        cols.append(col)
    return np.stack(cols, axis=1)


def targets_from_df(df):
    """NUEVO: predice dNeff_fo y dNeff_db directamente (columnas ya
    presentes en el CSV desde dataset_generator.py), en vez de Yp/log(D/H).
    No hace falta regenerar el dataset."""
    dNeff_fo = df["dNeff_fo"].values.astype(np.float64)
    dNeff_db = df["dNeff_db"].values.astype(np.float64)
    return np.stack([dNeff_fo, dNeff_db], axis=1)


class Standardizer:
    """Normalización (x-mean)/std, guardable/cargable como dict de numpy."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.std[self.std < 1e-12] = 1e-12  # evita división por cero

    @classmethod
    def fit(cls, X):
        return cls(mean=X.mean(axis=0), std=X.std(axis=0))

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, Xs):
        return Xs * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=d["mean"], std=d["std"])


class BBNDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ----------------------------------------------------------------------
# Arquitectura
# ----------------------------------------------------------------------
class BBNEmulatorMLP(nn.Module):
    def __init__(self, n_in, n_out=2, hidden_dim=128, n_layers=4, activation="silu"):
        super().__init__()
        act_cls = {"silu": nn.SiLU, "gelu": nn.GELU}[activation]

        layers = [nn.Linear(n_in, hidden_dim), act_cls()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers += [nn.Linear(hidden_dim, n_out)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------
def train_emulator(model_name, csv_path, epochs=500, batch_size=256, lr=1e-3,
                    hidden_dim=128, n_layers=4, activation="silu",
                    val_frac=0.1, patience=30, seed=0, out_path=None):

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    df = pd.read_csv(csv_path)
    df = df[df["valid"] == True].reset_index(drop=True)
    print(f"Dataset '{csv_path}': {len(df)} puntos válidos")

    X = params_to_features(df, model_name)
    Y = targets_from_df(df)

    x_scaler = Standardizer.fit(X)
    y_scaler = Standardizer.fit(Y)
    Xs = x_scaler.transform(X)
    Ys = y_scaler.transform(Y)

    full_ds = BBNDataset(Xs, Ys)
    n_val = int(len(full_ds) * val_frac)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BBNEmulatorMLP(
        n_in=X.shape[1], n_out=Y.shape[1],
        hidden_dim=hidden_dim, n_layers=n_layers, activation=activation
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d} | train_loss={train_loss:.3e} "
                  f"| val_loss={val_loss:.3e} | lr={current_lr:.2e}")

        if epochs_no_improve >= patience:
            print(f"Early stopping en epoch {epoch} "
                  f"(sin mejora en {patience} epochs)")
            break

    model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Validación en ESCALA FÍSICA (no en el espacio estandarizado/log):
    # esto es lo que realmente importa para el criterio de <0.1%.
    # ------------------------------------------------------------------
    from bbn_evaluator import DNEFF_OBS_ERR

    abs_err_fo, abs_err_db = _evaluate_physical_absolute_error(
        model, val_ds, x_scaler, y_scaler, device
    )

    print("\n=== Error absoluto en escala física (set de validación) ===")
    print(f"dNeff_fo: mediana={np.median(abs_err_fo):.5f}  máx={np.max(abs_err_fo):.5f}")
    print(f"dNeff_db: mediana={np.median(abs_err_db):.5f}  máx={np.max(abs_err_db):.5f}")

    # Objetivo: error << sigma observacional (DNEFF_OBS_ERR=0.070), para
    # que el chi2 calculado con el emulador no se distorsione.
    target_abs = 0.1 * DNEFF_OBS_ERR  # = 0.007
    ok_fo = np.median(abs_err_fo) < target_abs
    ok_db = np.median(abs_err_db) < target_abs
    if ok_fo and ok_db:
        print(f"OK: error mediano < {target_abs:.4f} (10% de sigma_obs={DNEFF_OBS_ERR}).")
    else:
        print(f"AVISO: no se alcanzó el objetivo de {target_abs:.4f}. "
              "Ver sugerencias habituales (más datos, red más grande, más epochs).")

    if out_path is None:
        out_path = f"emulator_{model_name}.pt"

    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "activation": activation,
        "n_in": X.shape[1],
        "n_out": Y.shape[1],
    }
    torch.save(checkpoint, out_path)
    print(f"\nCheckpoint guardado en: {out_path}")

    return model, x_scaler, y_scaler


def _evaluate_physical_absolute_error(model, val_ds, x_scaler, y_scaler, device):
    """
    Error ABSOLUTO (no relativo): con targets que pueden ser ~0 (dNeff en
    la meseta donde Gamma~0), el error relativo diverge sin que importe
    para el chi2 real, que compara contra DNEFF_OBS_ERR (una escala
    absoluta), no contra el valor verdadero.
    """
    model.eval()
    Xs_val = val_ds.dataset.X[val_ds.indices].numpy()
    Ys_val = val_ds.dataset.Y[val_ds.indices].numpy()
    with torch.no_grad():
        pred_s = model(torch.tensor(Xs_val, dtype=torch.float32).to(device)).cpu().numpy()
    pred_phys = y_scaler.inverse_transform(pred_s)
    true_phys = y_scaler.inverse_transform(Ys_val)
    abs_err_fo = np.abs(pred_phys[:, 0] - true_phys[:, 0])
    abs_err_db = np.abs(pred_phys[:, 1] - true_phys[:, 1])
    return abs_err_fo, abs_err_db

# ----------------------------------------------------------------------
# Inferencia: wrapper para usar como reemplazo del solver dentro del MCMC
# ----------------------------------------------------------------------
class BBNEmulator:
    """
    Carga un checkpoint y expone .predict(theta) -> (Yp, DH), donde theta
    es el vector de parámetros FÍSICOS (no transformados), en el mismo
    orden que MODEL_SPECS[model_name]['bounds'].keys().
    """

    def __init__(self, checkpoint_path, device=None):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model_name = ckpt["model_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BBNEmulatorMLP(
            n_in=ckpt["n_in"], n_out=ckpt["n_out"],
            hidden_dim=ckpt["hidden_dim"], n_layers=ckpt["n_layers"],
            activation=ckpt["activation"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.x_scaler = Standardizer.from_dict(ckpt["x_scaler"])
        self.y_scaler = Standardizer.from_dict(ckpt["y_scaler"])

    def predict(self, theta):
        """theta: array-like de parámetros físicos [alpha, Nf, (n)]."""
        spec = MODEL_SPECS[self.model_name]
        names = list(spec["bounds"].keys())

        theta = np.asarray(theta, dtype=np.float64)
        x = theta.copy()
        for i, name in enumerate(names):
            if spec["log_param"].get(name, False):
                x[i] = np.log10(x[i])

        xs = self.x_scaler.transform(x[None, :])
        with torch.no_grad():
            pred_s = self.model(
                torch.tensor(xs, dtype=torch.float32).to(self.device)
            ).cpu().numpy()

        pred_phys = self.y_scaler.inverse_transform(pred_s)[0]
        dNeff_fo_pred = float(pred_phys[0])
        dNeff_db_pred = float(pred_phys[1])
        return dNeff_fo_pred, dNeff_db_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    args = parser.parse_args()

    train_emulator(
        model_name=args.model, csv_path=args.csv,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    )