"""ITP 'final' .mat files -> one table of daily (00:00 UTC) positions.

    python preprocessing/02_itp_positions.py

Reads  data/raw/itp/itp{n}final.mat   (variables date [Y,M,D,h,...], lat, lon)
Writes data/intermediate/itp_daily_positions.nc  with itp, lat, lon, time

Only positions are used by the tutorial: the ITP tracks define *where and when* synthetic
observations are taken. (The 2025 version also computed drift velocities here; they were not used.)
"""
import numpy as np
import pandas as pd
import xarray as xr

from config import INTERMEDIATE, ITP_END, ITP_NUMBERS, RAW


def load_mat(path):
    """Return date (6, n), lat (n,), lon (n,) from a v7.3 or older MATLAB file."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return f["date"][:], f["lat"][:].ravel(), f["lon"][:].ravel()
    except OSError:
        from scipy.io import loadmat
        m = loadmat(path)
        return m["date"], m["lat"].ravel(), m["lon"].ravel()


def main():
    rows = []
    for n in ITP_NUMBERS:
        path = RAW / "itp" / f"itp{n}final.mat"
        if not path.exists():
            continue
        date, lat, lon = load_mat(path)
        ok = ~np.isnan(date).any(axis=0) & (date[3] == 0)          # complete timestamps at hour 0
        t = pd.to_datetime({"year": date[0, ok], "month": date[1, ok], "day": date[2, ok]})
        rows.append(pd.DataFrame({"itp": n, "lat": lat[ok], "lon": lon[ok], "time": t.values}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df.time <= ITP_END].sort_values(["itp", "time"]).reset_index(drop=True)

    ds = xr.Dataset({"itp": ("sample", df.itp.values), "lat": ("sample", df.lat.values),
                     "lon": ("sample", df.lon.values), "time": ("sample", df.time.values)})
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(INTERMEDIATE / "itp_daily_positions.nc")
    print(f"{len(df)} daily positions from {df.itp.nunique()} ITPs")


if __name__ == "__main__":
    main()
