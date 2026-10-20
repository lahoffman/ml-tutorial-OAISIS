"""The neural network: hyperparameters, build, train, reconstruct."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import data as D
from .metrics import corr_metric, nrmse_loss  # noqa: F401  (nrmse_loss is an alternative loss users can pick)

HP = dict(
    n_units=32,        # neurons per hidden layer
    n_layers=2,        # hidden layers
    dropout=0.0,       # dropout rate (0 = off)
    l2=1e-3,           # L2 weight penalty
    lr=1e-3,           # learning rate
    epochs=30,
    batch_size=256,
    activation="relu",
    loss="mse",        # or metrics.nrmse_loss
)


def build_model(n_in, n_out, n_units, n_layers, dropout, l2, lr, activation, loss, **_):
    model = keras.Sequential([layers.Input(shape=(n_in,))])
    for _ in range(n_layers):
        model.add(layers.Dense(n_units, activation=activation, kernel_regularizer=keras.regularizers.l2(l2)))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(n_out))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=[corr_metric])
    return model


def train_model(x_tr, y_tr, x_va, y_va, hp=HP, verbose=0, seed=0):
    """Build and fit a network; returns (model, history dict)."""
    keras.utils.set_random_seed(seed)
    model = build_model(x_tr.shape[1], y_tr.shape[1], **hp)
    hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=hp["epochs"],
                     batch_size=hp["batch_size"], shuffle=True, verbose=verbose)
    return model, hist.history


def train(p=None, hp=HP, verbose=0, seed=0):
    """Convenience: train on a `data.prepared()` namespace (all ITP samples by default)."""
    p = D.prepared() if p is None else p
    return train_model(p.x_tr, p.y_tr, p.x_va, p.y_va, hp=hp, verbose=verbose, seed=seed)


def predict(model, scaler, X):
    """Physical-unit predictions for a table X in physical units."""
    return scaler.y_inv(model.predict(scaler.x(X), verbose=0))


def reconstruct(model, scaler):
    """Apply a trained network to the full-Arctic inputs -> (ny, nx, nt, 2) ice velocity; NaN over land."""
    m = D.maps
    X = D.maps_X(scaler.features)
    pred = np.full((X.shape[0], 2), np.nan, dtype="float32")
    pred[m.ok] = predict(model, scaler, X[m.ok])
    return pred.reshape(m.ny, m.nx, m.nt, 2)


def linear_baseline(p, verbose=True):
    """Multiple linear regression on the same inputs: the score any network has to beat.
    Returns (model, validation mse in scaled units) — the mse is the dashed line in the learning curve."""
    from sklearn.linear_model import LinearRegression
    from .metrics import report
    linear = LinearRegression().fit(p.x_tr, p.y_tr)
    if verbose:
        report(p.Y_te, p.scaler.y_inv(linear.predict(p.x_te)), "linear")
    return linear, float(np.mean((linear.predict(p.x_va) - p.y_va) ** 2))


def n_parameters(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))
