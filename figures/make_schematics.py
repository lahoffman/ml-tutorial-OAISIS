"""Draw the two schematics at the top of the notebook, from the tutorial's own data and two small networks.

    python figures/make_schematics.py          (about a minute on CPU)

figures/training.png        inputs at ITP locations -> neural network -> outputs at ITP locations,
                            for one ITP (NN 1, ITP 95) and the same ITP plus a second one elsewhere (NN 2, ITPs 95 + 5)
figures/reconstruction.png  full-Arctic inputs -> NN 1 / NN 2 -> predicted maps, scored against the truth,
                            and the skill of the two compared
figures/sparse_to_global.png one network trained on all ITP samples, then applied to the full grid (no comparison yet)
figures/problem.png         the problem: wind known everywhere, ice velocity known only at the instruments -> learn a -> b
"""
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cartopy.crs as ccrs
import cmocean
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
ds = xr.open_dataset(ROOT / "data" / "oaisis_tutorial_IPSL.nc")
PROJ = ccrs.NorthPolarStereo(central_longitude=-90)
INK, LAND, OCEAN, LAVENDER, ORANGE, GREY = "#222222", "#d9d9d9", "#eef3f7", "#b4a7d6", "#f5a742", "#9a9a9a"
plt.rcParams.update({"font.family": "serif", "font.size": 11})

# ---------------------------------------------------------------- data & the two observing systems
lat, lon, track = ds.obs_lat.values, ds.obs_lon.values, ds.obs_track.values
doy = ds.obs_time.dt.dayofyear.values
ids, n_per = np.unique(track, return_counts=True)
NN1_TRACKS = [95]                                                         # NN 1: one ITP (two months north of the Laptev Sea)
NN2_TRACKS = [95, 5]                                                      # NN 2: the same ITP plus ITP 5 (six months toward Fram Strait)
obs_sets = {"NN 1": np.isin(track, NN1_TRACKS), "NN 2": np.isin(track, NN2_TRACKS), "NN": np.ones(track.size, bool)}

MEMBERS = list(ds.member.values)
TRAIN, VAL = MEMBERS[1:21], MEMBERS[21:26]


def table(mem, sel=slice(None), maps=False):
    if maps:
        g = lambda v: ds["map_" + v].values
        geo = (ds.map_time.dt.dayofyear.values[None, None, :], ds.map_lat.values[:, :, None], ds.map_lon.values[:, :, None])
    else:
        g = lambda v: ds["obs_" + v].sel(member=mem).values[:, sel]
        geo = (doy[None, sel], lat[None, sel], lon[None, sel])
    arrs = np.broadcast_arrays(g("uas"), g("vas"), g("sic"), *geo[:2], np.cos(np.radians(geo[2])), np.sin(np.radians(geo[2])))
    X = np.stack([a.ravel() for a in arrs], 1).astype("float32")
    Y = np.stack([a.ravel() for a in np.broadcast_arrays(g("siu"), g("siv"))], 1).astype("float32")
    return X, Y


def fit(sel, seed=0):
    Xtr, Ytr = table(TRAIN, sel); Xva, Yva = table(VAL, sel)
    mu, sd = Xtr.mean(0), Xtr.std(0); mu[2:] = 0; sd[2:] = [100, 365, 90, 1, 1]
    ymu, ysd = Ytr.mean(0), Ytr.std(0)
    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(1e-3)                                     # same architecture and settings as the notebook
    m = keras.Sequential([layers.Input((7,)), layers.Dense(32, activation="relu", kernel_regularizer=reg),
                          layers.Dense(32, activation="relu", kernel_regularizer=reg), layers.Dense(2)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    m.fit((Xtr - mu) / sd, (Ytr - ymu) / ysd, validation_data=((Xva - mu) / sd, (Yva - ymu) / ysd), epochs=15, batch_size=256, verbose=0)
    return lambda X: m.predict((X - mu) / sd, verbose=0) * ysd + ymu


X_map, Y_map = table(None, maps=True)
ok = np.isfinite(X_map).all(1)
ny, nx, nt = ds.map_siu.shape
ice = np.isfinite(ds.map_siu.values).all(2) & (np.nanmean(ds.map_sic.values, 2) > 15)
truth = Y_map.reshape(ny, nx, nt, 2)


def vskill(yt, yp):
    err = np.nansum((yp - yt) ** 2, -1); var = np.nansum((yt - np.nanmean(yt, 0)) ** 2, -1)
    return 1 - np.sqrt(np.nanmean(err)) / np.sqrt(np.nanmean(var))


recon, score = {}, {}
for name, sel in obs_sets.items():
    print(name, sel.sum(), "samples")
    predict = fit(sel)
    pred = np.full_like(Y_map, np.nan); pred[ok] = predict(X_map[ok])
    recon[name] = pred.reshape(ny, nx, nt, 2)
    score[name] = vskill(truth[ice].reshape(-1, 2), recon[name][ice].reshape(-1, 2))
    print("   skill %.3f" % score[name])


# ---------------------------------------------------------------- drawing helpers
def polar(ax, lat_min=70):
    x, y = PROJ.transform_point(0, lat_min, ccrs.PlateCarree())
    r = np.hypot(x, y)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.set_boundary(mpath.Path(np.column_stack([np.sin(th), np.cos(th)]) * 0.5 + 0.5), transform=ax.transAxes)
    ax.set_facecolor(LAND); ax.spines["geo"].set_edgecolor(INK); ax.spines["geo"].set_linewidth(0.8)
    # ocean
    f = np.where(np.isfinite(ds.map_sic.values[:, :, 0]), 1.0, np.nan)
    mesh(ax, f, ListedColormap([OCEAN]), 0, 2)


def mesh(ax, f, cmap, vmin, vmax):
    lo, la = ds.map_lon.values, ds.map_lat.values
    n = lo.shape[1]; perm = (n - 2 - np.arange(n)) % n
    lo, la, f = (np.vstack([a, a[-1:, perm]]) for a in (lo, la, f))
    xyz = PROJ.transform_points(ccrs.PlateCarree(), lo, la)
    return ax.pcolormesh(xyz[..., 0], xyz[..., 1], np.ma.masked_invalid(f), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")


def points(ax, sel, c, cmap=None, vmin=None, vmax=None, **kw):
    return ax.scatter(lon[sel], lat[sel], c=c, s=9, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), linewidths=0, zorder=3, **kw)


def frame(fig, ax, title, fontsize=12):
    """Slide-9 style panel: black header with the title, thin frame around the axes."""
    b = ax.get_position()
    pad, head = 0.012, 0.055
    fig.add_artist(Rectangle((b.x0 - pad, b.y0 - pad), b.width + 2 * pad, b.height + 2 * pad + head, transform=fig.transFigure,
                             fc="white", ec=INK, lw=1.2, zorder=0))
    fig.add_artist(Rectangle((b.x0 - pad, b.y1 + pad), b.width + 2 * pad, head, transform=fig.transFigure, fc=INK, ec=INK, zorder=1))
    fig.text(b.x0 + b.width / 2, b.y1 + pad + head / 2, title, ha="center", va="center", color="white", fontsize=fontsize, fontweight="bold")


def network(fig, ax, label):
    """Black rounded box with a small network inside and a label (slide-9 'machine learning model' box)."""
    ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(-0.6, 4.6); ax.set_aspect("equal")
    b = ax.get_position()
    fig.add_artist(FancyBboxPatch((b.x0, b.y0), b.width, b.height, boxstyle="round,pad=0.01,rounding_size=0.02",
                                  transform=fig.transFigure, fc=INK, ec=INK, zorder=0))
    lay = [np.linspace(0.8, 3.2, 3), np.linspace(0.2, 3.8, 4), np.linspace(1.4, 2.6, 2)]
    for x, (ya, yb) in enumerate(zip(lay[:-1], lay[1:])):
        for a in ya:
            for c in yb:
                ax.plot([x + 0.5, x + 1.5], [a, c], color="white", lw=0.6, alpha=0.6, zorder=1)
    for x, ys in enumerate(lay):
        for y in ys:
            ax.add_patch(Circle((x + 0.5, y), 0.16, fc="white", ec="white", zorder=2))
    ax.text(1.5, -0.35, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold")


def badge(fig, x, y, n, color=INK):
    """Circled step number drawn by hand (font-independent)."""
    w, h = fig.get_size_inches()
    fig.add_artist(Ellipse((x, y), 0.034 * h / w, 0.034, transform=fig.transFigure, fc="white", ec=color, lw=1.8, zorder=5))
    fig.text(x, y, str(n), ha="center", va="center", fontsize=14, color=color, fontweight="bold", zorder=6)


def arrow(fig, a, b, y=None, dx=0.0):
    pa, pb = a.get_position(), b.get_position()
    y = y if y is not None else (pa.y0 + pa.y1) / 2
    fig.add_artist(FancyArrowPatch((pa.x1 + 0.016 + dx, y), (pb.x0 - 0.008, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=20, color=INK, lw=2))


# ---------------------------------------------------------------- figure 1: training
fig = plt.figure(figsize=(12, 9.2))
gs = fig.add_gridspec(2, 3, width_ratios=[4, 1.6, 4], wspace=0.55, hspace=0.75, left=0.08, right=0.97, top=0.92, bottom=0.07)
wind_speed = np.hypot(ds.obs_uas.sel(member="r1i1p1f1").values, ds.obs_vas.sel(member="r1i1p1f1").values)
ice_speed = np.hypot(ds.obs_siu.sel(member="r1i1p1f1").values, ds.obs_siv.sel(member="r1i1p1f1").values)
for row, (name, sel) in enumerate(list(obs_sets.items())[:2]):
    ax_in = fig.add_subplot(gs[row, 0], projection=PROJ); polar(ax_in)
    ax_nn = fig.add_subplot(gs[row, 1])
    ax_out = fig.add_subplot(gs[row, 2], projection=PROJ); polar(ax_out)
    points(ax_in, sel, wind_speed[sel], cmocean.cm.tempo, 0, 12)
    points(ax_out, sel, ice_speed[sel], cmocean.cm.speed, 0, 0.3)
    if row == 1:                                                           # halo under what is new relative to NN 1
        new = sel & ~obs_sets["NN 1"]
        for ax in (ax_in, ax_out):
            ax.scatter(lon[new], lat[new], s=60, color=LAVENDER, alpha=0.25, linewidths=0, transform=ccrs.PlateCarree(), zorder=2)
    frame(fig, ax_in, "inputs"); frame(fig, ax_out, "outputs")
    network(fig, ax_nn, name)
    arrow(fig, ax_in, ax_nn); arrow(fig, ax_nn, ax_out)
    bi, bo = ax_in.get_position(), ax_out.get_position()
    fig.text(bi.x0 + bi.width / 2, bi.y0 - 0.03, "wind · ice concentration · position · date", ha="center", va="top", fontsize=10, style="italic")
    fig.text(bo.x0 + bo.width / 2, bo.y0 - 0.03, "sea-ice velocity", ha="center", va="top", fontsize=10, style="italic")
    badge(fig, 0.035, (bi.y0 + bi.y1) / 2, row + 1)
# "add observations" arrow between the two steps, in the left margin
b0, b1 = fig.axes[0].get_position(), fig.axes[3].get_position()
y0, y1 = (b0.y0 + b0.y1) / 2 - 0.03, (b1.y0 + b1.y1) / 2 + 0.03
fig.add_artist(FancyArrowPatch((0.035, y0), (0.035, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color=LAVENDER, lw=2.5))
fig.text(0.052, (y0 + y1) / 2, "+", color=LAVENDER, fontsize=28, fontweight="bold", va="center", ha="left")
fig.savefig(ROOT / "figures" / "training.png", dpi=160, facecolor="white")
print("written figures/training.png")

# ---------------------------------------------------------------- figure 2: reconstruction
fig = plt.figure(figsize=(17.5, 9.2))
gs = fig.add_gridspec(2, 6, width_ratios=[4, 1.6, 4, 0.9, 4, 3.6], wspace=0.5, hspace=0.75, left=0.05, right=0.98, top=0.92, bottom=0.07)
wind_map = np.hypot(ds.map_uas.values[:, :, 0], ds.map_vas.values[:, :, 0])
truth_speed = np.hypot(truth[:, :, 0, 0], truth[:, :, 0, 1])
for row, name in enumerate(list(obs_sets)[:2]):
    ax_in = fig.add_subplot(gs[row, 0], projection=PROJ); polar(ax_in, 66)
    ax_nn = fig.add_subplot(gs[row, 1])
    ax_pr = fig.add_subplot(gs[row, 2], projection=PROJ); polar(ax_pr, 66)
    ax_tr = fig.add_subplot(gs[row, 4], projection=PROJ); polar(ax_tr, 66)
    mesh(ax_in, wind_map, cmocean.cm.tempo, 0, 12)
    pr = np.hypot(recon[name][:, :, 0, 0], recon[name][:, :, 0, 1])
    mesh(ax_pr, np.where(ice, pr, np.nan), cmocean.cm.speed, 0, 0.3)
    mesh(ax_tr, np.where(ice, truth_speed, np.nan), cmocean.cm.speed, 0, 0.3)
    frame(fig, ax_in, "inputs"); frame(fig, ax_pr, "predicted outputs"); frame(fig, ax_tr, "true outputs")
    network(fig, ax_nn, name)
    arrow(fig, ax_in, ax_nn); arrow(fig, ax_nn, ax_pr)
    # "vs." between predicted and true
    bp, bt = ax_pr.get_position(), ax_tr.get_position()
    fig.text((bp.x1 + bt.x0) / 2, (bp.y0 + bp.y1) / 2, "vs.", ha="center", va="center", fontsize=14, style="italic")
    fig.text(bp.x0 + bp.width / 2, bp.y0 - 0.03, f"skill {score[name]:.2f}", ha="center", va="top", fontsize=12, fontweight="bold")
    badge(fig, 0.02, (bp.y0 + bp.y1) / 2, row + 1)
# step 3: compare
axb = fig.add_subplot(gs[:, 5])
names = list(obs_sets)[:2]
axb.bar([0, 1], [score[n] for n in names], color=[GREY, LAVENDER], width=0.6)
axb.set_xticks([0, 1]); axb.set_xticklabels(names, fontsize=12)
axb.set_ylim(0, max(0.7, max(score.values()) + 0.1)); axb.set_ylabel("reconstruction skill", fontsize=12)
for sp in ("top", "right"): axb.spines[sp].set_visible(False)
bb = axb.get_position(); axb.set_position([bb.x0 + 0.045, bb.y0 + 0.12, bb.width - 0.045, bb.height - 0.24])
bb = axb.get_position()
badge(fig, bb.x0 + bb.width / 2, bb.y1 + 0.06, 3)
for row in range(2):
    bt = fig.axes[row * 4 + 3].get_position()
    yy = (bt.y0 + bt.y1) / 2
    fig.add_artist(FancyArrowPatch((bt.x1 + 0.016, yy), (bb.x0 - 0.045, yy), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=20, color=INK, lw=2))
fig.savefig(ROOT / "figures" / "reconstruction.png", dpi=160, facecolor="white")
print("written figures/reconstruction.png")

# ---------------------------------------------------------------- figure 3: one network, sparse -> global
fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 5, width_ratios=[4, 1.6, 4, 0.9, 4], wspace=0.5, hspace=0.85, left=0.06, right=0.96, top=0.90, bottom=0.07)
sel = obs_sets["NN"]
# row 0: training at the observation locations
ax_in = fig.add_subplot(gs[0, 0], projection=PROJ); polar(ax_in, 66)
ax_nn0 = fig.add_subplot(gs[0, 1])
ax_out = fig.add_subplot(gs[0, 2], projection=PROJ); polar(ax_out, 66)
points(ax_in, sel, wind_speed[sel], cmocean.cm.tempo, 0, 12)
points(ax_out, sel, ice_speed[sel], cmocean.cm.speed, 0, 0.3)
frame(fig, ax_in, "inputs at the ITP locations"); frame(fig, ax_out, "outputs at the ITP locations")
network(fig, ax_nn0, "NN"); arrow(fig, ax_in, ax_nn0); arrow(fig, ax_nn0, ax_out)
bi, bo = ax_in.get_position(), ax_out.get_position()
fig.text(bi.x0 + bi.width / 2, bi.y0 - 0.03, "wind · ice concentration · position · date", ha="center", va="top", fontsize=10, style="italic")
fig.text(bo.x0 + bo.width / 2, bo.y0 - 0.03, "sea-ice velocity (the training targets)", ha="center", va="top", fontsize=10, style="italic")
badge(fig, 0.025, (bi.y0 + bi.y1) / 2, 1)
fig.text(bi.x0 - 0.012, bi.y1 + 0.085, "train at the observation locations", ha="left", fontsize=13, fontweight="bold")
# row 1: the same network applied to the full grid
ax_min = fig.add_subplot(gs[1, 0], projection=PROJ); polar(ax_min, 66)
ax_nn1 = fig.add_subplot(gs[1, 1])
ax_pr = fig.add_subplot(gs[1, 2], projection=PROJ); polar(ax_pr, 66)
ax_tr = fig.add_subplot(gs[1, 4], projection=PROJ); polar(ax_tr, 66)
mesh(ax_min, wind_map, cmocean.cm.tempo, 0, 12)
pr = np.hypot(recon["NN"][:, :, 0, 0], recon["NN"][:, :, 0, 1])
mesh(ax_pr, np.where(ice, pr, np.nan), cmocean.cm.speed, 0, 0.3)
mesh(ax_tr, np.where(ice, truth_speed, np.nan), cmocean.cm.speed, 0, 0.3)
frame(fig, ax_min, "inputs everywhere"); frame(fig, ax_pr, "predicted outputs"); frame(fig, ax_tr, "true outputs")
network(fig, ax_nn1, "same NN"); arrow(fig, ax_min, ax_nn1); arrow(fig, ax_nn1, ax_pr)
bp, bt = ax_pr.get_position(), ax_tr.get_position()
fig.text((bp.x1 + bt.x0) / 2, (bp.y0 + bp.y1) / 2, "vs.", ha="center", va="center", fontsize=14, style="italic")
fig.text(bp.x0 + bp.width / 2, bp.y0 - 0.03, f"skill {score['NN']:.2f}", ha="center", va="top", fontsize=12, fontweight="bold")
bm = ax_min.get_position()
fig.text(bm.x0 + bm.width / 2, bm.y0 - 0.03, "the same inputs, on the whole grid", ha="center", va="top", fontsize=10, style="italic")
fig.text(bt.x0 + bt.width / 2, bt.y0 - 0.03, "known: the 'world' is a climate model", ha="center", va="top", fontsize=10, style="italic")
badge(fig, 0.025, (bp.y0 + bp.y1) / 2, 2)
fig.text(bm.x0 - 0.012, bm.y1 + 0.085, "reconstruct everywhere, and score against the truth", ha="left", fontsize=13, fontweight="bold")
# the trained weights are carried from the training box to the reconstruction box
n0, n1 = ax_nn0.get_position(), ax_nn1.get_position()
xm = (n0.x0 + n0.x1) / 2
fig.add_artist(FancyArrowPatch((xm, n0.y0 - 0.015), (xm, n1.y1 + 0.015), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color=LAVENDER, lw=2.5))
fig.text(xm + 0.012, (n0.y0 + n1.y1) / 2, "trained\nweights", color="#5b4b8a", fontsize=10, va="center", ha="left")
fig.savefig(ROOT / "figures" / "sparse_to_global.png", dpi=160, facecolor="white")
print("written figures/sparse_to_global.png")

# ---------------------------------------------------------------- figure 4: the problem
fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 3, width_ratios=[4, 2.2, 4], wspace=0.55, hspace=0.6, left=0.06, right=0.96, top=0.90, bottom=0.08)
ax_a = fig.add_subplot(gs[0, 0], projection=PROJ); polar(ax_a, 66)
ax_b = fig.add_subplot(gs[1, 0], projection=PROJ); polar(ax_b, 66)
ax_q = fig.add_subplot(gs[:, 1])
_tmp = fig.add_subplot(gs[:, 2]); _b = _tmp.get_position(); _tmp.remove()
_a = ax_a.get_position()
ax_r = fig.add_axes([_b.x0, (_b.y0 + _b.y1) / 2 - _a.height / 2, _a.width, _a.height], projection=PROJ); polar(ax_r, 66)
mesh(ax_a, wind_map, cmocean.cm.tempo, 0, 12)
points(ax_b, obs_sets["NN"], ice_speed[obs_sets["NN"]], cmocean.cm.speed, 0, 0.3)
mesh(ax_r, np.where(ice, truth_speed, np.nan), cmocean.cm.speed, 0, 0.3)
ax_r.add_patch(Circle((0.5, 0.5), 0.5, transform=ax_r.transAxes, fc="white", ec="none", alpha=0.72, zorder=5))
ax_r.text(0.5, 0.5, "?", transform=ax_r.transAxes, ha="center", va="center", fontsize=110, fontweight="bold", color=INK, zorder=6)
frame(fig, ax_a, "wind — known everywhere"); frame(fig, ax_b, "ice velocity — at instruments only", fontsize=10.5)
frame(fig, ax_r, "ice velocity — everywhere?")
# the relation to learn: a dashed box with a -> b
ax_q.axis("off"); ax_q.set_xlim(0, 1); ax_q.set_ylim(0, 1)
bq = ax_q.get_position()
fig.add_artist(FancyBboxPatch((bq.x0, bq.y0 + 0.3), bq.width, bq.height - 0.6, boxstyle="round,pad=0.01,rounding_size=0.02",
                              transform=fig.transFigure, fc="white", ec=INK, lw=1.6, ls=(0, (6, 4)), zorder=0))
fig.text(bq.x0 + bq.width / 2, bq.y0 + bq.height / 2 + 0.02, "a  →  b", ha="center", va="center", fontsize=22, fontweight="bold")
fig.text(bq.x0 + bq.width / 2, bq.y0 + bq.height / 2 - 0.045, "learn the relation", ha="center", va="center", fontsize=11, style="italic")
ba, bb, br = ax_a.get_position(), ax_b.get_position(), ax_r.get_position()
ym = bq.y0 + bq.height / 2
for b_ in (ba, bb):
    y_ = (b_.y0 + b_.y1) / 2
    fig.add_artist(FancyArrowPatch((b_.x1 + 0.016, y_), (bq.x0 - 0.008, ym + (0.04 if b_ is ba else -0.04)), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=20, color=INK, lw=2, connectionstyle="arc3,rad=" + ("-0.25" if b_ is ba else "0.25")))
fig.add_artist(FancyArrowPatch((bq.x1 + 0.012, ym), (br.x0 - 0.008, ym), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20, color=INK, lw=2))
fig.text(ba.x0 - 0.03, (ba.y0 + ba.y1) / 2, "a", fontsize=20, fontweight="bold", ha="center", va="center")
fig.text(bb.x0 - 0.03, (bb.y0 + bb.y1) / 2, "b", fontsize=20, fontweight="bold", ha="center", va="center")
fig.text((bq.x1 + br.x0) / 2, ym + 0.025, "apply it wherever\na is known", ha="center", va="bottom", fontsize=11, style="italic")
fig.savefig(ROOT / "figures" / "problem.png", dpi=160, facecolor="white")
print("written figures/problem.png")
