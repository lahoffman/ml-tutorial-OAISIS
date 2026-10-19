"""Arctic maps of the same five variables from one IPSL-CM6A-LR member (the reconstruction truth).

    python preprocessing/04_subset_ipsl_maps.py

Reads  data/raw/ipsl/<var>/..._{MAP_MEMBER}_...
Writes data/intermediate/maps_ipsl.nc  with (y, x, time) arrays on the ORCA1 ocean grid

Steps: select MAP_START..MAP_END, keep cells north of ARCTIC_LAT_MIN, drop the ORCA halo cells,
rotate siu/siv to east/north, regrid the 10 m wind onto the ocean grid by nearest neighbour.
"""
import numpy as np
import xarray as xr

from config import ARCTIC_LAT_MIN, FILL_THRESHOLD, INTERMEDIATE, MAP_END, MAP_MEMBER, MAP_START, RAW, ipsl_url
from utils import clean_fill, fill_fold_row, grid_angle, nearest_cells, rotate_to_east_north, trim_orca_halo


def open_var(var):
    _, name = ipsl_url(var, MAP_MEMBER)
    return xr.open_dataset(RAW / "ipsl" / var / name, use_cftime=True).sel(time=slice(MAP_START, MAP_END))


def main():
    ice = {v: trim_orca_halo(open_var(v), ydim="y", xdim="x") for v in ("siu", "siv", "siconc")}
    first_row = int(np.argmax((ice["siu"].nav_lat.values >= ARCTIC_LAT_MIN).any(axis=1)))
    ice = {v: d.isel(y=slice(first_row, None)) for v, d in ice.items()}   # drop rows entirely south of the cut
    lat, lon = ice["siu"].nav_lat.values, ice["siu"].nav_lon.values
    theta = grid_angle(lon, lat)
    arctic = lat >= ARCTIC_LAT_MIN

    def field(da):                                       # (time, y, x) -> (y, x, time), land/fill -> NaN
        a = clean_fill(da.values, FILL_THRESHOLD).transpose(1, 2, 0)
        a[~arctic] = np.nan
        return a

    siu, siv = rotate_to_east_north(field(ice["siu"].siu), field(ice["siv"].siv), theta[:, :, None])
    siu, siv = fill_fold_row(siu).astype("float32"), fill_fold_row(siv).astype("float32")
    sic = field(ice["siconc"].siconc)

    # wind: regular lat/lon grid -> nearest atmosphere cell for each ocean cell
    atm = {v: open_var(v) for v in ("uas", "vas")}
    alat, alon = np.meshgrid(atm["uas"].lat.values, ((atm["uas"].lon.values + 180) % 360) - 180, indexing="ij")
    flat, _ = nearest_cells(alat, alon, lat.ravel(), lon.ravel())
    ay, ax = np.unravel_index(flat, alat.shape)
    wind = {}
    for v in ("uas", "vas"):
        a = atm[v][v].values[:, ay, ax].T.reshape(lat.shape + (-1,)).astype("float32")
        a[~arctic] = np.nan
        wind[v] = a

    ds = xr.Dataset(
        {"map_uas": (("y", "x", "time"), wind["uas"]), "map_vas": (("y", "x", "time"), wind["vas"]),
         "map_sic": (("y", "x", "time"), sic), "map_siu": (("y", "x", "time"), siu), "map_siv": (("y", "x", "time"), siv),
         "map_grid_angle": (("y", "x"), np.degrees(theta).astype("float32"))},
        coords={"map_lat": (("y", "x"), lat), "map_lon": (("y", "x"), lon),
                "map_time": ("time", ice["siu"].time.values)})
    ds.to_netcdf(INTERMEDIATE / "maps_ipsl.nc")
    print(ds)


if __name__ == "__main__":
    main()
