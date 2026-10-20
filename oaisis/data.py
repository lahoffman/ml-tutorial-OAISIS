"""The dataset, and how it becomes a machine-learning table.

Everything is loaded when the module is imported; the tutorial file is 7 MB.

    ds                      the xarray Dataset (data/oaisis_tutorial_IPSL.nc)
    MEMBERS                 ensemble members 'r1i1p1f1' ... 'r31i1p1f1'
    lat_obs, lon_obs, ...   1-D arrays along the ITP samples
    FEATURES, TARGETS       the inputs and outputs of the network
    make_table, obs_table   arrays -> X (n, 7), Y (n, 2)
    SPLIT                   train / validation / test ensemble members
    Scaler                  standardization with training statistics
    prepared()              the three scaled tables and the scaler, ready for training
    maps                    the full-Arctic fields (X_map, Y_true_map, ice_mask, ...)
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

FILENAME = "data/oaisis_tutorial_IPSL.nc"


def find_data():
    """The tutorial NetCDF: $OAISIS_DATA, or data/oaisis_tutorial_IPSL.nc in the working directory,
    any of its parents, or next to this package."""
    candidates = [Path(os.environ["OAISIS_DATA"])] if "OAISIS_DATA" in os.environ else []
    cwd = Path.cwd().resolve()
    candidates += [p / FILENAME for p in [cwd, *cwd.parents]]
    candidates += [Path(__file__).resolve().parents[1] / FILENAME]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"{FILENAME} not found from {cwd}; set OAISIS_DATA to its path")


DATA = find_data()
ds = xr.open_dataset(DATA)

# --- the synthetic observations (ITP samples) -----------------------------------------------------
MEMBERS = list(ds.member.values)
lat_obs = ds.obs_lat.values
lon_obs = ds.obs_lon.values
time_obs = ds.obs_time.values
doy_obs = ds.obs_time.dt.dayofyear.values
year_obs = ds.obs_time.dt.year.values
month_obs = ds.obs_time.dt.month.values
track_obs = ds.obs_track.values
track_ids, track_len = np.unique(track_obs, return_counts=True)
n_samples = lat_obs.size

# --- the machine-learning table -------------------------------------------------------------------
FEATURES = ["uas", "vas", "sic", "doy", "lat", "cos_lon", "sin_lon"]
TARGETS = ["siu", "siv"]


def make_table(uas, vas, sic, doy, lat, lon, siu=None, siv=None):
    """Stack broadcastable arrays into X (n, 7) and Y (n, 2). Y is None if the targets are missing."""
    arrs = np.broadcast_arrays(uas, vas, sic, doy, lat, np.cos(np.radians(lon)), np.sin(np.radians(lon)))
    X = np.stack([a.ravel() for a in arrs], axis=1).astype("float32")
    Y = None
    if siu is not None:
        siu, siv = np.broadcast_arrays(siu, siv)
        Y = np.stack([siu.ravel(), siv.ravel()], axis=1).astype("float32")
    return X, Y


def obs_table(members, sample_mask=None):
    """Feature/target table for the given ensemble members, optionally for a subset of the ITP samples."""
    sel = slice(None) if sample_mask is None else sample_mask
    g = lambda v: ds["obs_" + v].sel(member=members).values[:, sel]      # (n_members, n_samples)
    return make_table(g("uas"), g("vas"), g("sic"), doy_obs[None, sel], lat_obs[None, sel], lon_obs[None, sel], g("siu"), g("siv"))


# split by ensemble member: r1 is kept for testing because the maps are from r1
SPLIT = {"train": MEMBERS[1:21], "val": MEMBERS[21:26], "test": [MEMBERS[0]] + MEMBERS[26:31]}


class Scaler:
    """Standardize features/targets with training statistics. Fixed scaling for sic, doy, lat, lon."""
    fixed = {"sic": 100.0, "doy": 365.0, "lat": 90.0, "cos_lon": 1.0, "sin_lon": 1.0}

    def fit(self, X, Y):
        self.x_mean, self.x_std = X.mean(0), X.std(0)
        self.y_mean, self.y_std = Y.mean(0), Y.std(0)
        for name, s in self.fixed.items():          # geometric inputs: divide by a constant
            i = FEATURES.index(name)
            self.x_mean[i], self.x_std[i] = 0.0, s
        return self

    def x(self, X):  return (X - self.x_mean) / self.x_std
    def y(self, Y):  return (Y - self.y_mean) / self.y_std
    def y_inv(self, Z): return Z * self.y_std + self.y_mean


def prepared(sample_mask=None):
    """Scaled train/val/test tables for a selection of ITP samples (all of them by default).
    Returns a namespace with X_*, Y_* (physical units), x_*, y_* (scaled) and `scaler`."""
    p = SimpleNamespace()
    for split, mem in SPLIT.items():
        X, Y = obs_table(mem, sample_mask)
        setattr(p, "X_" + split[:2], X); setattr(p, "Y_" + split[:2], Y)
    p.scaler = Scaler().fit(p.X_tr, p.Y_tr)
    for split in ("tr", "va", "te"):
        setattr(p, "x_" + split, p.scaler.x(getattr(p, "X_" + split)))
        setattr(p, "y_" + split, p.scaler.y(getattr(p, "Y_" + split)))
    return p


# --- the maps ---------------------------------------------------------------------------------------
maps = SimpleNamespace()
maps.lat, maps.lon = ds.map_lat.values, ds.map_lon.values
maps.time = ds.map_time.values
maps.doy = ds.map_time.dt.dayofyear.values
maps.sic, maps.uas, maps.vas, maps.siu, maps.siv = (ds["map_" + v].values for v in ("sic", "uas", "vas", "siu", "siv"))
maps.ny, maps.nx, maps.nt = maps.siu.shape
with np.errstate(all="ignore"):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        maps.ice_mask = np.isfinite(maps.siu).all(axis=2) & (np.nanmean(maps.sic, axis=2) > 15)   # where a reconstruction is meaningful
        maps.land = ~np.isfinite(maps.sic).any(axis=2) & (maps.lat > 55)                           # the model's land mask
maps.X, maps.Y = make_table(maps.uas, maps.vas, maps.sic, maps.doy[None, None, :], maps.lat[:, :, None], maps.lon[:, :, None], maps.siu, maps.siv)
maps.ok = np.isfinite(maps.X).all(axis=1)                                 # land cells have NaN inputs
maps.Y_true = maps.Y.reshape(maps.ny, maps.nx, maps.nt, 2)


def day_label(t):
    return str(maps.time[t])[:10]
