"""Select subsets of the ITP samples: each function returns a boolean mask over the samples.
Combine with & (and), | (or), ~ (not)."""
import numpy as np

from . import data as D


def everything():
    return np.ones(D.n_samples, bool)


def tracks(*ids):
    """ITP track ids, e.g. tracks(95) or tracks(95, 5)."""
    return np.isin(D.track_obs, ids)


def region(lon=(-180, 180), lat=(-90, 90)):
    """A longitude range (going east from lon[0] to lon[1]) and a latitude range."""
    in_lon = ((D.lon_obs - lon[0]) % 360) <= ((lon[1] - lon[0]) % 360 or 360)
    return in_lon & (D.lat_obs >= lat[0]) & (D.lat_obs <= lat[1])


def near(lat, lon, radius_km=75):
    """All samples within radius_km of a point: a pseudo-mooring."""
    la0, lo0, la, lo = map(np.radians, (lat, lon, D.lat_obs, D.lon_obs))
    d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2))
    return d <= radius_km


def years(first, last=None):
    return (D.year_obs >= first) & (D.year_obs <= (last or first))


def months(*m):
    return np.isin(D.month_obs, m)


def season(name):
    return months(*{"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}[name])


def every(n):
    """Every n-th sample: a thinned observing system."""
    return np.arange(D.n_samples) % n == 0


def describe(sel):
    """One line: how many samples, tracks and years a selection covers."""
    if not sel.any():
        return "  empty selection"
    t = D.time_obs[sel]
    return f"{sel.sum():6d} samples, {len(np.unique(D.track_obs[sel])):3d} tracks, {str(t.min())[:4]}-{str(t.max())[:4]}"


def long_tracks(min_days=30):
    return D.track_ids[D.track_len >= min_days]
