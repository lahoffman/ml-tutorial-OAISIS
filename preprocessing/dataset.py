"""Attributes and writer for data/oaisis_tutorial_IPSL.nc (used by 05_build_dataset.py and 05b_...)."""
import datetime as _dt

import xarray as xr

ATTRS = {
    # synthetic observations along ITP tracks
    "obs_lat":  dict(long_name="latitude of ITP position", units="degrees_north"),
    "obs_lon":  dict(long_name="longitude of ITP position", units="degrees_east"),
    "obs_time": dict(long_name="date of ITP position"),
    "obs_track": dict(long_name="ITP track id", comment="consecutive samples from one drifting ITP share an id; "
                      "numbered in time order. From the 2025 files the real ITP numbers are not available, so tracks "
                      "are recovered from gaps in date/position (preprocessing/utils.segment_tracks)."),
    "obs_uas":  dict(long_name="eastward 10 m wind at ITP position", units="m s-1", standard_name="eastward_wind"),
    "obs_vas":  dict(long_name="northward 10 m wind at ITP position", units="m s-1", standard_name="northward_wind"),
    "obs_sic":  dict(long_name="sea-ice concentration at ITP position", units="%", standard_name="sea_ice_area_fraction"),
    "obs_siu":  dict(long_name="eastward sea-ice velocity at ITP position", units="m s-1", standard_name="eastward_sea_ice_velocity",
                     comment="rotated from the CMIP6 grid-x component using the local grid angle (see map_grid_angle)"),
    "obs_siv":  dict(long_name="northward sea-ice velocity at ITP position", units="m s-1", standard_name="northward_sea_ice_velocity",
                     comment="rotated from the CMIP6 grid-y component using the local grid angle (see map_grid_angle)"),
    "member":   dict(long_name="IPSL-CM6A-LR historical ensemble member (variant label)"),
    # Arctic maps
    "map_lat":  dict(long_name="latitude of ocean grid cell centre", units="degrees_north"),
    "map_lon":  dict(long_name="longitude of ocean grid cell centre", units="degrees_east"),
    "map_time": dict(long_name="date"),
    "map_uas":  dict(long_name="eastward 10 m wind (nearest atmosphere cell)", units="m s-1", standard_name="eastward_wind"),
    "map_vas":  dict(long_name="northward 10 m wind (nearest atmosphere cell)", units="m s-1", standard_name="northward_wind"),
    "map_sic":  dict(long_name="sea-ice concentration", units="%", standard_name="sea_ice_area_fraction"),
    "map_siu":  dict(long_name="eastward sea-ice velocity", units="m s-1", standard_name="eastward_sea_ice_velocity",
                     comment="rotated from the CMIP6 grid-x/y components; NaN over land and south of 55N"),
    "map_siv":  dict(long_name="northward sea-ice velocity", units="m s-1", standard_name="northward_sea_ice_velocity",
                     comment="rotated from the CMIP6 grid-x/y components; NaN over land and south of 55N"),
    "map_grid_angle": dict(long_name="angle of the ocean grid x axis, counter-clockwise from east", units="degrees",
                           comment="derived from map_lat/map_lon; u_east = u_x cos(a) - v_y sin(a), v_north = u_x sin(a) + v_y cos(a)"),
}

GLOBAL = dict(
    title="OAISIS ML tutorial: synthetic Arctic sea-ice observations and maps from IPSL-CM6A-LR",
    summary=("Two parts. (1) obs_*: daily output of 31 IPSL-CM6A-LR historical members sampled at the positions and "
             "dates of real Ice-Tethered Profiler (ITP) tracks, 2004-2014 (dims member x sample). "
             "(2) map_*: daily Arctic fields of the same variables from one member on the native ORCA1 ocean grid "
             "(dims y x x x time). All velocity components are geographic east/north."),
    source="CMIP6 CMIP historical, IPSL-CM6A-LR, daily (SIday: siu, siv, siconc; day: uas, vas). ITP positions: WHOI Ice-Tethered Profiler Program.",
    references=("Boucher et al. 2020 (IPSL-CM6A-LR); Toole et al. 2011, Krishfield et al. 2008 (ITP); "
                "Eyring et al. 2016 (CMIP6)"),
    conventions="units and standard_name follow CF where a name exists",
    processing="scripts in preprocessing/ of https://github.com/lahoffman/ml-tutorial-OAISIS",
)


def write_dataset(obs, maps, path, history):
    """Merge the obs and map datasets, attach attributes, write NetCDF with compression."""
    ds = xr.merge([obs, maps])
    for k, a in ATTRS.items():
        if k in ds:
            ds[k].attrs.update(a)
    ds.attrs.update(GLOBAL, history=f"{_dt.date.today()}: {history}")
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars if ds[v].dtype.kind == "f"}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=enc)
    return ds
