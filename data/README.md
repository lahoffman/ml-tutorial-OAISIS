# `oaisis_tutorial_IPSL.nc`

One NetCDF file, two parts, all from the CMIP6 **historical** experiment of **IPSL-CM6A-LR**.
Open with `xarray.open_dataset`; every variable carries `long_name`, `units` and, where useful, a `comment`.

## Part 1 — synthetic observations along ITP tracks (`obs_*`)

Dimensions `member` (31) × `sample` (11 242). For each daily position of a real Ice-Tethered Profiler
(2004-08-20 to 2014-12-31, 71–90 °N), the value of the model at the nearest ocean grid cell and the same
calendar day, for each of the 31 historical members `r1i1p1f1` … `r31i1p1f1`.

| variable | dims | units | description |
|---|---|---|---|
| `obs_lat`, `obs_lon` | sample | degrees | ITP position |
| `obs_time` | sample | datetime | ITP date (00 UTC fix) |
| `obs_track` | sample | — | ITP track id (1–105, in time order); 45 tracks have ≥ 30 days |
| `member` | member | — | variant label, e.g. `r1i1p1f1` |
| `obs_uas`, `obs_vas` | member, sample | m s⁻¹ | eastward / northward 10 m wind (`day/uas`, `day/vas`) |
| `obs_sic` | member, sample | % | sea-ice concentration (`SIday/siconc`) |
| `obs_siu`, `obs_siv` | member, sample | m s⁻¹ | **eastward / northward** sea-ice velocity (rotated from `SIday/siu`, `siv`) |

Samples are ordered by ITP and time; 96 % of consecutive samples are consecutive days of the same ITP.
Samples where any member was on land or missing were removed, so there are no fill values. The 2025 files did
not keep the WHOI ITP numbers, so `obs_track` is recovered from the data: a new track starts where the date
does not advance or the position jumps further than ice could drift (`preprocessing/utils.segment_tracks`).
Most tracks are single ITPs; an ITP with a long data gap may appear as two.

## Part 2 — Arctic maps from one member (`map_*`)

Dimensions `y` (65) × `x` (360) × `time` (10): member `r1i1p1f1`, 2000-11-25 to 2000-12-04, on the native
NEMO ORCA1 ocean grid north of ~50 °N (values are NaN south of 55 °N and over land). The duplicate "halo"
cells of the ORCA grid have been removed; the last row (`y = -1`) is the grid's north fold.

| variable | dims | units | description |
|---|---|---|---|
| `map_lat`, `map_lon` | y, x | degrees | cell centres |
| `map_time` | time | datetime | |
| `map_uas`, `map_vas` | y, x, time | m s⁻¹ | 10 m wind, nearest atmosphere cell |
| `map_sic` | y, x, time | % | sea-ice concentration |
| `map_siu`, `map_siv` | y, x, time | m s⁻¹ | **eastward / northward** sea-ice velocity |
| `map_grid_angle` | y, x | degrees | angle of the grid x axis from east, used for the rotation |

## Two things worth knowing

**Velocity components.** CMIP6 `siu`/`siv` are the components along the model grid's own x/y axes.
On IPSL's tripolar grid the x axis is rotated by tens of degrees from east over the Arctic. Both parts
of this file store *geographic* components: `u_east = u_x cos θ − v_y sin θ`, `v_north = u_x sin θ + v_y cos θ`,
with θ = `map_grid_angle` (for the ITP samples, θ of the nearest grid cell).

**Independence.** Member r1 appears in both parts. The tutorial keeps it out of training.

## Provenance

Built by `preprocessing/05b_build_dataset_from_zenodo.py` from the files of the 2025 tutorial
(Zenodo [10.5281/zenodo.15597587](https://doi.org/10.5281/zenodo.15597587)); the from-scratch pipeline
is `preprocessing/00`–`05`. The `history` global attribute records which.
