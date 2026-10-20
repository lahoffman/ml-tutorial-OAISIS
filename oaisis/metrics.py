"""Vector skill and vector correlation for the (u, v) ice velocity, in numpy and in Keras."""
import numpy as np
import tensorflow as tf

from . import data as D


def skill(y_true, y_pred, axis=0):
    """Vector skill: 1 - RMS(|error|) / RMS(|anomaly of the truth|). The (u, v) components are on the last axis;
    the statistics are taken along `axis`."""
    err = np.nansum((y_pred - y_true) ** 2, axis=-1)
    var = np.nansum((y_true - np.nanmean(y_true, axis=axis, keepdims=True)) ** 2, axis=-1)
    return 1.0 - np.sqrt(np.nanmean(err, axis=axis)) / np.sqrt(np.nanmean(var, axis=axis))


def corr(y_true, y_pred, axis=0):
    """Vector correlation (sum of the component covariances over the product of the vector standard deviations)."""
    a = y_true - np.nanmean(y_true, axis=axis, keepdims=True)
    b = y_pred - np.nanmean(y_pred, axis=axis, keepdims=True)
    cov = np.nansum(np.nansum(a * b, axis=-1), axis=axis)
    return cov / np.sqrt(np.nansum(np.nansum(a * a, axis=-1), axis=axis) * np.nansum(np.nansum(b * b, axis=-1), axis=axis))


def report(y_true, y_pred, label=""):
    s, c = skill(y_true, y_pred), corr(y_true, y_pred)
    print(f"{label:12s} skill {s:.3f}   corr {c:.3f}")
    return s, c


# Keras versions of the same quantities, used during training
def nrmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true))) / (tf.math.reduce_std(y_true) + 1e-7)


def corr_metric(y_true, y_pred):
    a = y_true - tf.reduce_mean(y_true); b = y_pred - tf.reduce_mean(y_pred)
    return tf.reduce_sum(a * b) / (tf.sqrt(tf.reduce_sum(a * a) * tf.reduce_sum(b * b)) + 1e-7)


# --- for maps ---------------------------------------------------------------------------------------
def pooled_scores(Y_rec, Y_true=None, mask=None, label="pooled"):
    """Skill and correlation over all ice-covered cells and days together."""
    Y_true = D.maps.Y_true if Y_true is None else Y_true
    mask = D.maps.ice_mask if mask is None else mask
    return report(Y_true[mask].reshape(-1, 2), Y_rec[mask].reshape(-1, 2), label)


def cell_scores(Y_rec, Y_true=None, mask=None, min_std=0.01):
    """Per-cell vector skill and correlation (ny, nx) over time, masked where the truth barely varies."""
    Y_true = D.maps.Y_true if Y_true is None else Y_true
    mask = D.maps.ice_mask if mask is None else mask
    s, c = skill(Y_true, Y_rec, axis=2), corr(Y_true, Y_rec, axis=2)
    anom = Y_true - np.nanmean(Y_true, axis=2, keepdims=True)
    low = (np.sqrt(np.nanmean(np.nansum(anom ** 2, axis=-1), axis=2)) < min_std) | ~mask
    s[low] = np.nan; c[low] = np.nan
    return s, c
