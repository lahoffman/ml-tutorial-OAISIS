"""Maps and plots. All maps are north-polar stereographic with a circular boundary."""
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import data as D

PROJ = ccrs.NorthPolarStereo(central_longitude=-90)
PC = ccrs.PlateCarree()
LAVENDER, DARK, GREY = "#b4a7d6", "#333333", "#9a9a9a"
_coastlines = None


def _have_coastlines():
    """Natural Earth coastlines need a download the first time; without network we fall back to the model's land mask."""
    global _coastlines
    if _coastlines is None:
        try:
            import cartopy.io.shapereader as shp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shp.natural_earth(resolution="110m", category="physical", name="coastline")
                shp.natural_earth(resolution="110m", category="physical", name="land")
            _coastlines = True
        except Exception:
            _coastlines = False
    return _coastlines


def polar_axes(ax, title="", lat_min=65):
    """North-polar map with a circular boundary. Limits are set in projected coordinates
    (set_extent with a full longitude range is unreliable across cartopy versions)."""
    x, y = ax.projection.transform_point(0, lat_min, PC)
    r = np.hypot(x, y)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.set_boundary(mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5), transform=ax.transAxes)
    if _have_coastlines():
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.coastlines(resolution="110m", linewidth=0.5)
    else:                                                   # offline: the climate model's own land cells
        polar_pcolormesh(ax, D.maps.lon, D.maps.lat, np.where(D.maps.land, 1.0, np.nan), cmap=ListedColormap(["lightgray"]), zorder=0)
    ax.set_title(title)
    return ax


def polar_pcolormesh(ax, lon2d, lat2d, field, fold=True, **kw):
    """pcolormesh on a curvilinear lat/lon grid, drawn in the map projection's own coordinates (no dateline streaks).
    fold=True appends the mirror image of the last row (ORCA north fold) so there is no gap along the fold line."""
    if fold:
        perm = (lon2d.shape[1] - 2 - np.arange(lon2d.shape[1])) % lon2d.shape[1]   # cell j <-> cell (n-2-j) across the fold
        lon2d, lat2d, field = (np.vstack([a, a[-1:, perm]]) for a in (lon2d, lat2d, field))
    xyz = ax.projection.transform_points(PC, lon2d, lat2d)
    return ax.pcolormesh(xyz[..., 0], xyz[..., 1], np.ma.masked_invalid(field), shading="auto", **kw)


def map_field(ax, field, title="", cmap=cmocean.cm.balance, ice_only=False, colorbar=True, lat_min=65, **kw):
    """One full-Arctic field (ny, nx) on a polar map."""
    polar_axes(ax, title, lat_min=lat_min)
    f = np.where(D.maps.ice_mask, field, np.nan) if ice_only else field
    pm = polar_pcolormesh(ax, D.maps.lon, D.maps.lat, f, cmap=cmap, **kw)
    if colorbar:
        plt.colorbar(pm, ax=ax, shrink=0.7)
    return pm


def _grid(n, size=4.4, height=None, **kw):
    fig, axs = plt.subplots(1, n, figsize=(size * n, height or size + 0.6), subplot_kw={"projection": PROJ}, layout="constrained", **kw)
    return fig, np.atleast_1d(axs)


# --- the data -------------------------------------------------------------------------------------------
def tracks(min_days=30, lat_min=72, label=True, figsize=(9, 9)):
    """All ITP tracks with at least `min_days` days, coloured and labelled by id."""
    ids = D.track_ids[D.track_len >= min_days]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": PROJ})
    polar_axes(ax, f"{len(ids)} ITP tracks with >= {min_days} days (of {len(D.track_ids)})", lat_min=lat_min)
    cmap = plt.get_cmap("tab20")
    for i in ids:
        m = D.track_obs == i
        ax.scatter(D.lon_obs[m], D.lat_obs[m], s=2, color=cmap(i % 20), transform=PC)
        if label:
            ax.text(D.lon_obs[m][0], D.lat_obs[m][0], str(i), fontsize=7, transform=PC, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    return ax


def obs_overview(member="r1i1p1f1"):
    """Where the ITPs were, coloured by date, and what the model says there: concentration, ice speed, wind speed."""
    m = D.ds.sel(member=member)
    speed_ice, speed_wind = np.hypot(m.obs_siu, m.obs_siv), np.hypot(m.obs_uas, m.obs_vas)
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), subplot_kw={"projection": PROJ}, layout="constrained")
    panels = [("day of year", D.doy_obs, cmocean.cm.phase, {}),
              ("ice concentration [%]", m.obs_sic, cmocean.cm.ice, {"vmin": 0, "vmax": 100}),
              ("ice speed [m/s]", speed_ice, cmocean.cm.speed, {"vmin": 0, "vmax": 0.4}),
              ("wind speed [m/s]", speed_wind, cmocean.cm.speed, {"vmin": 0, "vmax": 15})]
    for ax, (title, c, cmap, kw) in zip(axs.flat, panels):
        polar_axes(ax, title)
        sc = ax.scatter(D.lon_obs, D.lat_obs, c=c, s=4, cmap=cmap, transform=PC, **kw)
        plt.colorbar(sc, ax=ax, shrink=0.6)
    fig.suptitle(f"synthetic observations, member {member}")


def input_maps(t=0, member_label="r1i1p1f1"):
    """The full-Arctic fields for one day: the two targets, the three physical inputs, and where the ITPs sampled."""
    m = D.maps
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={"projection": PROJ}, layout="constrained")
    fields = [("siu — eastward ice velocity [m/s]", m.siu[:, :, t], cmocean.cm.balance, dict(vmin=-0.3, vmax=0.3)),
              ("siv — northward ice velocity [m/s]", m.siv[:, :, t], cmocean.cm.balance, dict(vmin=-0.3, vmax=0.3)),
              ("sic — ice concentration [%]", m.sic[:, :, t], cmocean.cm.ice, dict(vmin=0, vmax=100)),
              ("uas — eastward wind [m/s]", m.uas[:, :, t], cmocean.cm.balance, dict(vmin=-12, vmax=12)),
              ("vas — northward wind [m/s]", m.vas[:, :, t], cmocean.cm.balance, dict(vmin=-12, vmax=12)),
              ("where the ITPs sampled (2004-2014)", None, None, {})]
    for ax, (title, f, cmap, kw) in zip(axs.flat, fields):
        if f is None:
            polar_axes(ax, title)
            ax.scatter(D.lon_obs, D.lat_obs, s=1, c="k", transform=PC)
        else:
            map_field(ax, f, title, cmap=cmap, **kw)
    fig.suptitle(f"IPSL-CM6A-LR {member_label}, {D.day_label(t)}")


# --- training and evaluation ----------------------------------------------------------------------------
def history(h, baseline=None, ylim=(0, None)):
    """Learning curve: loss and correlation per epoch, training vs validation."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, k, lab in zip(axs, ["loss", "corr_metric"], ["loss", "correlation"]):
        ax.plot(h[k], label="train"); ax.plot(h["val_" + k], label="validation")
        ax.set_xlabel("epoch"); ax.set_ylabel(lab)
    if baseline is not None:
        axs[0].axhline(baseline, color="k", ls="--", lw=1, label="linear regression (validation)")
    axs[0].set_ylim(*ylim); axs[0].legend(); axs[1].legend()
    plt.tight_layout()
    return axs


def prediction_scatter(Y_true, Y_pred, title="", n_members=None):
    """Predicted vs true u and v (hexbin), and the vector error along the tracks for the first member."""
    from .metrics import skill, corr
    s, c = skill(Y_true, Y_pred), corr(Y_true, Y_pred)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, k, name in zip(axs, range(2), D.TARGETS):
        ax.hexbin(Y_true[:, k], Y_pred[:, k], gridsize=60, bins="log", cmap="Greys")
        ax.plot([-0.6, 0.6], [-0.6, 0.6], "r-", lw=1)
        ax.set_xlabel(f"true {name} [m/s]"); ax.set_ylabel(f"predicted {name} [m/s]"); ax.set_aspect("equal")
    n_members = n_members or Y_true.shape[0] // D.n_samples
    err = np.hypot(*(Y_pred - Y_true).reshape(n_members, -1, 2)[0].T)
    axs[2].remove(); ax = fig.add_subplot(1, 3, 3, projection=PROJ)
    polar_axes(ax, "|predicted - true| [m/s], one member")
    sc = ax.scatter(D.lon_obs, D.lat_obs, c=err, s=4, cmap=cmocean.cm.amp, vmin=0, vmax=0.1, transform=PC)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(f"{title}  skill {s:.2f}, r = {c:.2f}")


# --- reconstruction -------------------------------------------------------------------------------------
def reconstruction_day(Y_rec, t=0, vmax=0.3):
    """True speed, reconstructed speed and the magnitude of the vector error for one day."""
    Yt = D.maps.Y_true
    speed_true, speed_rec = np.hypot(Yt[:, :, t, 0], Yt[:, :, t, 1]), np.hypot(Y_rec[:, :, t, 0], Y_rec[:, :, t, 1])
    err = np.hypot(*(Y_rec[:, :, t] - Yt[:, :, t]).transpose(2, 0, 1))
    fig, axs = _grid(3, size=5)
    map_field(axs[0], speed_true, "true ice speed [m/s]", cmocean.cm.speed, ice_only=True, vmin=0, vmax=vmax)
    map_field(axs[1], speed_rec, "reconstructed ice speed [m/s]", cmocean.cm.speed, ice_only=True, vmin=0, vmax=vmax)
    map_field(axs[2], err, "|reconstruction - truth| [m/s]", cmocean.cm.amp, ice_only=True, vmin=0, vmax=vmax / 2)
    fig.suptitle(D.day_label(t))


def cell_scores(s, c, title=""):
    """Maps of per-cell skill and correlation."""
    fig, axs = _grid(2, size=5)
    map_field(axs[0], s, "skill", vmin=-1, vmax=1)
    map_field(axs[1], c, "correlation", vmin=-1, vmax=1)
    fig.suptitle(title)


# --- observing-system experiments -----------------------------------------------------------------------
def sampling(scenarios, title="", lat_min=70):
    """One map per observing system: its samples in black, the rest in grey."""
    fig, axs = _grid(len(scenarios), size=3.6, height=3.8)
    for ax, (label, sel) in zip(axs, scenarios.items()):
        polar_axes(ax, label, lat_min=lat_min)
        ax.scatter(D.lon_obs[~sel], D.lat_obs[~sel], s=1, c="lightgray", transform=PC)
        ax.scatter(D.lon_obs[sel], D.lat_obs[sel], s=1, c="k", transform=PC)
    fig.suptitle(title)


def skill_bars(results, title="", reference="all"):
    """Reconstruction skill per observing system (mean over seeds, min-max as error bars; sample count on each bar)."""
    names = list(results)
    mean = np.array([results[n]["skills"].mean() for n in names])
    lo = mean - np.array([results[n]["skills"].min() for n in names]); hi = np.array([results[n]["skills"].max() for n in names]) - mean
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4))
    ax.bar(names, mean, yerr=[lo, hi], capsize=4, color=[DARK if n == reference else LAVENDER for n in names], width=0.6)
    for i, n in enumerate(names):
        ax.text(i, 0.02, f"{results[n]['n_obs']:,}", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_ylabel("reconstruction skill"); ax.set_ylim(0, max(0.8, mean.max() + 0.1))
    ax.set_title(f"{title}   (number of samples on each bar)", fontsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ax


def skill_difference(results, reference="all", vmax=0.5):
    """Skill map of the reference minus skill map of each scenario: red = the missing observations were helping there."""
    base = results[reference]["skill_map"]
    others = [r for r in results.values() if r["name"] != reference]
    fig, axs = _grid(len(others), size=4.4, height=5)
    for ax, r in zip(axs, others):
        map_field(ax, base - r["skill_map"], f"{reference} - {r['name']}", vmin=-vmax, vmax=vmax)
    fig.suptitle("skill of the reference minus skill of the scenario: red = the missing observations were helping there", fontsize=10)


def skill_curve(steps, ref, title="", labels={}):
    """Skill against the number of ITPs for a sequence of growing observing systems."""
    n = np.arange(1, len(steps) + 1)
    mean = np.array([s["skills"].mean() for s in steps])
    lo = np.array([s["skills"].min() for s in steps]); hi = np.array([s["skills"].max() for s in steps])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(n, lo, hi, color=LAVENDER, alpha=0.4, lw=0, label="min-max over seeds")
    ax.plot(n, mean, "o-", color="#5b4b8a", label="skill (seed mean)")
    ax.axhline(ref["skills"].mean(), color=DARK, ls="--", lw=1.2, label=f"all ITP samples ({ref['n_obs']:,})")
    for k, lab in labels.items():
        ax.annotate(lab, (k, mean[k - 1]), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=10)
    ax.set_xticks(n); ax.set_xlabel("number of ITPs used for training"); ax.set_ylabel("reconstruction skill")
    ax.set_ylim(min(0.35, lo.min() - 0.05), 0.65); ax.legend(loc="lower right", frameon=False); ax.set_title(title, fontsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ax
