"""Build data/oaisis_tutorial_IPSL.nc from the 2025 Zenodo files (10.5281/zenodo.15597587).

    python preprocessing/05b_build_dataset_from_zenodo.py <folder containing the two .nc files>

Input:  RECONSTRUCTION_MASK.nc   (nm=51 members x nt=11242 ITP samples; rows 0-30 are IPSL-CM6A-LR r1-r31)
        RECONSTRUCTIONS_MAPS.nc  (332 x 362 x 10 days, IPSL-CM6A-LR r1, lat/lon/date tiled over time)
Output: data/oaisis_tutorial_IPSL.nc  (same layout as 05_build_dataset.py)

The Zenodo files came out of the 2025 version of scripts 02-05. This script applies, without
re-downloading CMIP6, the corrections that are now built into 03 and 04:
  * keep IPSL-CM6A-LR only (drop the CanESM5 and MIROC6 rows),
  * rotate siu/siv from grid x/y to east/north (ITP samples: angle of the nearest grid cell),
  * drop the ORCA halo cells and fill the half-empty fold row,
  * store lat/lon/time once, with proper time coordinates and attributes.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from config import ARCTIC_LAT_MIN, DATASET, FILL_THRESHOLD, MEMBERS
from dataset import write_dataset
from utils import clean_fill, fill_fold_row, grid_angle, nearest_cells, rotate_to_east_north, trim_orca_halo

REF = datetime(1850, 1, 1)


def main(folder):
    folder = Path(folder).expanduser()
    mask = xr.open_dataset(folder / "RECONSTRUCTION_MASK.nc")
    maps = trim_orca_halo(xr.open_dataset(folder / "RECONSTRUCTIONS_MAPS.nc"))
    first_row = int(np.argmax((maps.lat.values[:, :, 0] >= ARCTIC_LAT_MIN).any(axis=1)))
    maps = maps.isel(nx=slice(first_row, None))                 # drop rows entirely south of the Arctic cut

    # --- maps ----------------------------------------------------------------------------------
    lat, lon = maps.lat.values[:, :, 0].astype(float), maps.lon.values[:, :, 0].astype(float)
    theta = grid_angle(lon, lat)
    f = lambda v: clean_fill(maps[v].values, FILL_THRESHOLD)
    siu, siv = rotate_to_east_north(f("siu_IPSL"), f("siv_IPSL"), theta[:, :, None])
    siu, siv = fill_fold_row(siu).astype("float32"), fill_fold_row(siv).astype("float32")
    time = np.array([REF + timedelta(days=float(d)) for d in maps.date.values[0, 0, :]], dtype="datetime64[ns]")
    time = time.astype("datetime64[D]").astype("datetime64[ns]")        # midday stamps -> dates
    map_ds = xr.Dataset(
        {"map_uas": (("y", "x", "time"), f("uas_IPSL")), "map_vas": (("y", "x", "time"), f("vas_IPSL")),
         "map_sic": (("y", "x", "time"), f("sic_IPSL")), "map_siu": (("y", "x", "time"), siu), "map_siv": (("y", "x", "time"), siv),
         "map_grid_angle": (("y", "x"), np.degrees(theta).astype("float32"))},
        coords={"map_lat": (("y", "x"), lat), "map_lon": (("y", "x"), lon), "map_time": ("time", time)})

    # --- ITP samples (IPSL rows only) ----------------------------------------------------------
    n = len(MEMBERS)
    olat, olon = mask.lat.values, mask.lon.values
    flat, _ = nearest_cells(lat, lon, olat, olon, valid=np.isfinite(theta) & (lat > 60))
    theta_obs = theta.ravel()[flat]
    osiu, osiv = rotate_to_east_north(mask.siu_CMIP6.values[:n], mask.siv_CMIP6.values[:n], theta_obs[None, :])
    otime = np.array([REF + timedelta(days=int(d)) for d in mask.date.values], dtype="datetime64[ns]")
    obs_ds = xr.Dataset(
        {"obs_uas": (("member", "sample"), mask.uas_CMIP6.values[:n]), "obs_vas": (("member", "sample"), mask.vas_CMIP6.values[:n]),
         "obs_sic": (("member", "sample"), mask.sic_CMIP6.values[:n]),
         "obs_siu": (("member", "sample"), osiu.astype("float32")), "obs_siv": (("member", "sample"), osiv.astype("float32"))},
        coords={"member": MEMBERS, "obs_lat": ("sample", olat), "obs_lon": ("sample", olon), "obs_time": ("sample", otime)})

    ds = write_dataset(obs_ds, map_ds, DATASET,
                       "built with preprocessing/05b_build_dataset_from_zenodo.py from Zenodo 10.5281/zenodo.15597587 "
                       "(IPSL rows only; velocities rotated to east/north; ORCA halo removed)")
    print(ds)
    print(f"\nwritten {DATASET}  ({DATASET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
