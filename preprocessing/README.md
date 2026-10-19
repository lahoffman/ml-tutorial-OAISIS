# Preprocessing

How `data/oaisis_tutorial_IPSL.nc` is made. **You do not need to run any of this to do the tutorial.**

```
00_download_itp.py       WHOI ITP 'final' products            -> data/raw/itp/itp{n}final.mat
01_download_ipsl.py      IPSL-CM6A-LR daily output from ESGF  -> data/raw/ipsl/<var>/*.nc
02_itp_positions.py      daily (00 UTC) ITP positions          -> data/intermediate/itp_daily_positions.nc
03_colocate_itp_ipsl.py  model sampled at ITP positions/dates  -> data/intermediate/colocated_ipsl.nc
04_subset_ipsl_maps.py   Arctic maps from one member           -> data/intermediate/maps_ipsl.nc
05_build_dataset.py      combine + attributes                  -> data/oaisis_tutorial_IPSL.nc

05b_build_dataset_from_zenodo.py   the 2025 Zenodo files -> data/oaisis_tutorial_IPSL.nc   (what was actually run)

config.py   paths, members, dates, URLs — the only place these live
utils.py    co-location, grid angle / rotation, ORCA halo handling, calendar-aware date matching
dataset.py  variable attributes and the NetCDF writer shared by 05 and 05b
```

Run from the repository root, e.g. `python preprocessing/05b_build_dataset_from_zenodo.py ~/Downloads`.

## What each step does

**02 — ITP positions.** Each ITP reports a GPS fix several times a day; we keep the 00 UTC fix, giving one
position per ITP per day. Only positions are used: the ITP tracks tell us *where and when* an observing
system was sampling, and the model provides the values.

**03 — co-location.** For every position/day and every ensemble member: the nearest ocean grid cell
(great-circle distance on the 2-D `nav_lat`/`nav_lon`), the nearest model day (matched on calendar dates
with `cftime`, so it is correct for any CMIP6 calendar), and the model value there. Wind comes from the
regular atmosphere grid with its own nearest-cell lookup. Sea-ice velocity is rotated from the grid's x/y
axes to east/north *before* sampling (see below). Samples where any member/variable is land or fill are
dropped at the end, so all members share the same sample axis.

**04 — maps.** One member, a time window, cells north of 55 °N. The two periodic halo columns and the
north-fold row of the ORCA grid are removed; the ice velocity is rotated; the wind is put on the ocean
grid by nearest neighbour.

**05 / 05b — dataset.** Merge, attach CF-style attributes, compress. `05b` does the same starting from the
two files of the 2025 tutorial (`RECONSTRUCTION_MASK.nc`, `RECONSTRUCTIONS_MAPS.nc` on Zenodo), keeping
only the IPSL-CM6A-LR rows and applying the rotation and halo handling after the fact.

## The grid-angle rotation

CMIP6 `siu`/`siv` have standard names `sea_ice_x_velocity` / `sea_ice_y_velocity`: components along the
model grid, not east/north. IPSL-CM6A-LR uses the NEMO tripolar ORCA1 grid, whose x axis is rotated by a
median of ~49° from east at the ITP positions and reverses across the fold between the two northern grid
poles. The wind, from the atmosphere model, *is* east/north. `utils.grid_angle` estimates the local angle
θ of the x axis from the gradient of the cell-centre lat/lon along the grid, and `rotate_to_east_north`
applies

    u_east  = u_x cos θ − v_y sin θ
    v_north = u_x sin θ + v_y cos θ

θ is stored in the dataset as `map_grid_angle`. The effect is large: without the rotation the wind–ice
correlation in the maps is 0.59 / 0.52 (u / v); with it, 0.72 / 0.88.

## Known limitations of the shipped data

* The maps cover only ten days (Nov–Dec 2000). Per-cell statistics over 10 days are noisy; the notebook
  says so. `MAP_START`/`MAP_END` in `config.py` control the window for a rebuild.
* The ITP positions in the Zenodo files were matched to model days by day number rather than calendar
  date. That is exact for IPSL-CM6A-LR (Gregorian calendar) but would be wrong for 365-day-calendar
  models, which is why 03 now matches on dates.
* ITP identifiers were not carried through the 2025 files. `obs_track` is therefore recovered from gaps in
  date/position (`utils.segment_tracks`, 105 tracks); a from-scratch build (03) stores the real ITP numbers.
