"""Sample IPSL-CM6A-LR daily fields at the ITP positions and dates (synthetic observations).

    python preprocessing/03_colocate_itp_ipsl.py

Reads  data/intermediate/itp_daily_positions.nc and data/raw/ipsl/<var>/...
Writes data/intermediate/colocated_ipsl.nc  with (member, sample) arrays of uas, vas, sic, siu, siv

For every ITP position/day and every member: nearest ocean-grid cell (haversine), nearest model day
(matched on calendar dates, so this works for any CMIP6 calendar), value at that cell/day.
Sea-ice velocity is rotated from the grid's x/y axes to east/north *before* sampling, so the
stored siu/siv are geographic components, like the wind.
Samples where any member/variable is land or fill are dropped at the end.
"""
import numpy as np
import xarray as xr

from config import FILL_THRESHOLD, INTERMEDIATE, IPSL_FILES, MEMBERS, RAW, ipsl_url
from utils import clean_fill, grid_angle, match_dates, nearest_cells, rotate_to_east_north

VARS = ["uas", "vas", "siconc", "siu", "siv"]


def open_var(var, member):
    _, name = ipsl_url(var, member)
    return xr.open_dataset(RAW / "ipsl" / var / name, use_cftime=True)


def sample(da, it, iy, ix):
    """da (time, y, x) at paired indices -> (n,)"""
    return da.values[it, iy, ix]


def main():
    itp = xr.open_dataset(INTERMEDIATE / "itp_daily_positions.nc")
    lat, lon, time = itp.lat.values, itp.lon.values, itp.time.to_index()
    out = {v: [] for v in VARS}

    for member in MEMBERS:
        print(member)
        ds = {v: open_var(v, member) for v in VARS}

        # ocean grid (curvilinear): one nearest-cell lookup shared by siconc/siu/siv
        glat, glon = ds["siu"].nav_lat.values, ds["siu"].nav_lon.values
        flat, _ = nearest_cells(glat, glon, lat, lon)
        iy, ix = np.unravel_index(flat, glat.shape)
        theta = grid_angle(glon, glat)[iy, ix]
        it_ocean = match_dates(ds["siu"].time, time)

        # atmosphere grid (regular): separate lookup
        alat, alon = np.meshgrid(ds["uas"].lat.values, ((ds["uas"].lon.values + 180) % 360) - 180, indexing="ij")
        aflat, _ = nearest_cells(alat, alon, lat, lon)
        ay, ax = np.unravel_index(aflat, alat.shape)
        it_atm = match_dates(ds["uas"].time, time)

        siu = clean_fill(sample(ds["siu"].siu, it_ocean, iy, ix), FILL_THRESHOLD)
        siv = clean_fill(sample(ds["siv"].siv, it_ocean, iy, ix), FILL_THRESHOLD)
        siu, siv = rotate_to_east_north(siu, siv, theta)
        out["siu"].append(siu); out["siv"].append(siv)
        out["siconc"].append(clean_fill(sample(ds["siconc"].siconc, it_ocean, iy, ix), FILL_THRESHOLD))
        out["uas"].append(sample(ds["uas"].uas, it_atm, ay, ax))
        out["vas"].append(sample(ds["vas"].vas, it_atm, ay, ax))

    arrs = {v: np.array(out[v], dtype="float32") for v in VARS}
    keep = np.all([np.isfinite(a).all(axis=0) for a in arrs.values()], axis=0)
    res = xr.Dataset({"obs_" + ("sic" if v == "siconc" else v): (("member", "sample"), arrs[v][:, keep]) for v in VARS},
                     coords={"member": MEMBERS, "obs_lat": ("sample", lat[keep]), "obs_lon": ("sample", lon[keep]),
                             "obs_time": ("sample", itp.time.values[keep]), "obs_track": ("sample", itp.itp.values[keep].astype("int16"))})
    res.to_netcdf(INTERMEDIATE / "colocated_ipsl.nc")
    print(f"kept {keep.sum()} of {keep.size} samples")


if __name__ == "__main__":
    main()
