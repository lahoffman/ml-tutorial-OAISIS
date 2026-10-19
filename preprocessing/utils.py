"""Shared functions for the preprocessing pipeline.

The notebook re-implements grid_angle / rotate_to_east_north inline so it stays self-contained;
this module is the reference implementation.
"""
import numpy as np
from scipy.spatial import cKDTree


# --- geometry -------------------------------------------------------------------------------------
def to_xyz(lat, lon):
    """Unit-sphere coordinates (n, 3) from degrees."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def nearest_cells(grid_lat, grid_lon, lat, lon, valid=None):
    """Index (into the flattened grid) of the nearest grid cell to each (lat, lon).

    Args:
        grid_lat, grid_lon: 2-D arrays.
        lat, lon: 1-D arrays of target points.
        valid: optional 2-D bool mask of usable cells.
    Returns:
        flat indices into grid_lat (n,), great-circle distance in km (n,).
    """
    valid = np.ones(grid_lat.shape, bool) if valid is None else valid
    flat = np.flatnonzero(valid)
    tree = cKDTree(to_xyz(grid_lat.ravel()[flat], grid_lon.ravel()[flat]))
    d, i = tree.query(to_xyz(lat, lon))
    return flat[i], 2 * 6371.0 * np.arcsin(np.clip(d / 2, 0, 1))


def grid_angle(lon2d, lat2d):
    """Angle (radians, counter-clockwise from east) of the grid x axis (axis=1) at every cell."""
    dlon = np.gradient(np.unwrap(np.radians(lon2d), axis=1), axis=1)
    dlat = np.gradient(np.radians(lat2d), axis=1)
    return np.arctan2(dlat, dlon * np.cos(np.radians(lat2d)))


def rotate_to_east_north(u, v, theta):
    """Rotate grid-relative (x, y) components to (eastward, northward)."""
    return u * np.cos(theta) - v * np.sin(theta), u * np.sin(theta) + v * np.cos(theta)


# --- ORCA grid housekeeping ------------------------------------------------------------------------
def trim_orca_halo(ds, ydim="nx", xdim="ny"):
    """Drop the periodic halo columns (first, last) and the north-fold row (last) of an ORCA grid."""
    return ds.isel({ydim: slice(0, -1), xdim: slice(1, -1)})


def fill_fold_row(field, axis_y=0):
    """On the trimmed grid's last row every point appears twice (cell j and cell n-2-j).
    NEMO stores some variables in only one of the pair; copy them to the twin."""
    field = np.moveaxis(field, axis_y, 0)
    n = field.shape[1]
    perm = (n - 2 - np.arange(n)) % n
    last = field[-1]
    field[-1] = np.where(np.isfinite(last), last, last[perm])
    return np.moveaxis(field, 0, axis_y)


def clean_fill(a, threshold=1e19):
    """CMIP6 fill values -> NaN."""
    a = np.asarray(a, dtype="float32").copy()
    a[np.abs(a) > threshold] = np.nan
    return a


# --- ITP tracks ---------------------------------------------------------------------------------------
def segment_tracks(time, lat, lon, max_km_per_day=30.0, slack_km=50.0):
    """Split time-ordered samples into tracks: a new track starts where the date does not advance,
    or where the position jump is larger than drift could explain over the elapsed days.
    Returns 1-based integer ids (n,)."""
    days = np.asarray(time, dtype="datetime64[D]").astype(int)
    dt = np.diff(days)
    la, lo = np.radians(lat), np.radians(lon)
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2))
    new = (dt <= 0) | (d > slack_km + max_km_per_day * np.maximum(dt, 0))
    return np.concatenate([[1], 1 + np.cumsum(new)]).astype("int16")


# --- time -------------------------------------------------------------------------------------------
def match_dates(model_time, target_dates):
    """Index of the model time step closest to each target date.

    Args:
        model_time: xarray DataArray of the model's time coordinate (decoded with its own calendar).
        target_dates: array of numpy datetime64 or cftime dates.
    """
    import cftime
    cal = getattr(model_time.values[0], "calendar", "standard")
    units = "days since 1850-01-01"
    m = cftime.date2num(model_time.values, units, cal)
    t = np.array([cftime.date2num(cftime.datetime(d.year, d.month, d.day, calendar=cal), units, cal)
                  for d in target_dates])
    return np.abs(m[None, :] - t[:, None]).argmin(axis=1)
