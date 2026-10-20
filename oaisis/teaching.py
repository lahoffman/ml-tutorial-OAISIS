"""Figures that explain the machine-learning concepts with the tutorial's own data
(the ice velocity / wind relation along the ITP tracks)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon, Rectangle

from . import data as D
from .selectors import tracks

INK, LAVENDER, PURPLE, ORANGE, BLUE, GREY = "#222222", "#b4a7d6", "#5b4b8a", "#f5a742", "#4f81d6", "#9a9a9a"


# --- one input, one output --------------------------------------------------------------------------------
def xy(member="r1i1p1f1", x="uas", y="siu", sel=None, n=None, seed=0):
    """The simplest possible version of the problem: one input (x) and one output (y) along the ITP tracks."""
    m = D.ds.sel(member=member)
    xs, ys = m["obs_" + x].values, m["obs_" + y].values
    if sel is not None:
        xs, ys = xs[sel], ys[sel]
    if n is not None and n < xs.size:
        i = np.random.default_rng(seed).choice(xs.size, n, replace=False); xs, ys = xs[i], ys[i]
    return xs, ys


def fit_line(xs, ys):
    """Least-squares w and b for y = w x + b."""
    w, b = np.polyfit(xs, ys, 1)
    return w, b


def mse(xs, ys, w, b):
    return np.mean((w * xs + b - ys) ** 2)


def scatter(xs, ys, model=None, residuals=0, ax=None, xlabel="eastward wind, uas [m/s]", ylabel="eastward ice velocity, siu [m/s]", title=""):
    """Scatter of y against x; `model=(w, b)` adds the line; `residuals=k` draws the error of k points."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(xs, ys, s=8, color=ORANGE, alpha=0.7, lw=0, label="data (one ITP sample each)")
    if model is not None:
        w, b = model
        xx = np.array([xs.min(), xs.max()])
        ax.plot(xx, w * xx + b, color=PURPLE, lw=2, label=f"model  y = {w:.3f} x {b:+.3f}")
        if residuals:
            idx = np.random.default_rng(1).choice(xs.size, residuals, replace=False)
            for i in idx:
                ax.annotate("", (xs[i], w * xs[i] + b), (xs[i], ys[i]), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
            ax.scatter(xs[idx], ys[idx], s=14, color=INK, zorder=3)
            ax.text(0.98, 0.04, "error $e_i$ = model − truth", transform=ax.transAxes, ha="right", fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ax


def loss_vs_weight(xs, ys, b=None, w_range=None, ax=None, marks=()):
    """The loss (mean squared error) as a function of the weight w, with the intercept b held fixed."""
    w_opt, b_opt = fit_line(xs, ys)
    b = b_opt if b is None else b
    w_range = (w_opt - 0.05, w_opt + 0.05) if w_range is None else w_range
    ws = np.linspace(*w_range, 200)
    J = [mse(xs, ys, w, b) for w in ws]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ws, J, color=PURPLE, lw=2)
    ax.plot(w_opt, mse(xs, ys, w_opt, b), "o", color=INK, ms=7)
    ax.annotate(f"minimum: w = {w_opt:.3f}", (w_opt, mse(xs, ys, w_opt, b)), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=12)
    for w in marks:
        ax.plot(w, mse(xs, ys, w, b), "o", color=ORANGE, ms=6)
    ax.set_xlabel("weight w"); ax.set_ylabel("loss J(w) = mean squared error [m²/s²]")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ax, (w_opt, b)


def gradient_descent(xs, ys, lr=1.0, w0=None, steps=15, b=None, ax=None, title=None):
    """Minimize J(w) by gradient descent: w <- w - lr * dJ/dw. Returns the sequence of weights.
    (dJ/dw = 2 mean(x (w x + b - y)); lr is in units that make lr=1 a sensible step for this problem.)"""
    w_opt, b_opt = fit_line(xs, ys)
    b = b_opt if b is None else b
    w0 = w_opt + 0.045 if w0 is None else w0
    scale = 1.0 / (2 * np.mean(xs ** 2))                   # lr = 1 would reach the minimum in one step for pure least squares
    ws = [w0]
    for _ in range(steps):
        grad = 2 * np.mean(xs * (ws[-1] * xs + b - ys))
        ws.append(ws[-1] - lr * scale * grad)
    ws = np.array(ws)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    span = max(0.05, np.abs(ws - w_opt).max() * 1.1)
    loss_vs_weight(xs, ys, b=b, w_range=(w_opt - span, w_opt + span), ax=ax)
    J = np.array([mse(xs, ys, w, b) for w in ws])
    ax.plot(ws, J, "-", color=ORANGE, lw=1, alpha=0.8)
    ax.scatter(ws, J, c=np.arange(len(ws)), cmap="Oranges_r", s=40, zorder=3, edgecolor=INK, lw=0.5)
    ax.set_title(title if title is not None else f"learning rate {lr:g}: {steps} steps, final w = {ws[-1]:.3f}", fontsize=12)
    return ws


def gradient_descent_panel(xs, ys, lrs=(0.1, 0.8, 2.1), steps=15):
    """Too small, about right, too large: the same descent with three learning rates."""
    fig, axs = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 4.2), layout="constrained")
    for ax, lr, lab in zip(axs, lrs, ("too small", "about right", "too large")):
        gradient_descent(xs, ys, lr=lr, steps=steps, ax=ax, title=f"learning rate {lr:g} — {lab}")
    return axs


# --- what moves the ice -----------------------------------------------------------------------------------
def forces():
    """Side view of sea ice on the ocean: pushed by the wind above, dragged by the current below, resisted by
    neighbouring floes. The arrows are the terms of the momentum balance."""
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(-2.2, 2.6); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 12, 2.6, fc="#eef4f9", ec="none"))                       # air
    ax.add_patch(Rectangle((0, -2.2), 12, 2.2, fc="#3b6e9e", ec="none"))                      # ocean
    for y, a in ((-0.5, 0.25), (-1.1, 0.18), (-1.7, 0.12)):                                   # a little texture in the water
        ax.plot(np.linspace(0, 12, 300), y + 0.05 * np.sin(np.linspace(0, 12, 300) * 3), color="white", lw=1, alpha=a)
    floes = [(0.6, 2.4, 0.45), (3.3, 3.2, 0.6), (6.8, 2.3, 0.5), (9.4, 2.1, 0.4)]            # x, width, thickness
    for x, w, h in floes:
        pts = [(x, -0.15 * h), (x + 0.12 * w, -h), (x + 0.88 * w, -h), (x + w, -0.1 * h), (x + 0.93 * w, 0.75 * h), (x + 0.1 * w, 0.8 * h)]
        ax.add_patch(Polygon(pts, closed=True, fc="white", ec="#9fb3c8", lw=1.5, zorder=3))
    kw = dict(mutation_scale=28, lw=0, zorder=5)
    ax.add_patch(FancyArrowPatch((2.6, 1.9), (6.6, 1.9), arrowstyle="simple,head_length=0.9,head_width=1.2,tail_width=0.55", fc="#5b7fa6", **kw))
    ax.text(4.6, 2.25, "wind", ha="center", va="bottom", fontsize=15, color="#2f4f6f")
    ax.add_patch(FancyArrowPatch((3.0, -1.35), (5.6, -1.35), arrowstyle="simple,head_length=0.8,head_width=1.0,tail_width=0.4", fc="#a9c6e3", **kw))
    ax.text(4.3, -1.75, "ocean current", ha="center", va="top", fontsize=15, color="white")
    ax.add_patch(FancyArrowPatch((4.3, 0.45), (6.2, 0.45), arrowstyle="simple,head_length=0.8,head_width=1.0,tail_width=0.4", fc=INK, **kw))
    ax.text(5.25, 0.95, "ice motion", ha="center", va="bottom", fontsize=15, color=INK, fontweight="bold")
    ax.add_patch(FancyArrowPatch((7.5, 0.25), (6.75, 0.25), arrowstyle="simple,head_length=0.6,head_width=0.8,tail_width=0.3", fc=ORANGE, **kw))
    ax.add_patch(FancyArrowPatch((6.1, 0.25), (6.75, 0.25), arrowstyle="simple,head_length=0.6,head_width=0.8,tail_width=0.3", fc=ORANGE, **kw))
    ax.text(6.8, -0.75, "floes pushing on each other", ha="center", va="top", fontsize=13, color=ORANGE)
    ax.annotate("", (10.4, 0.6), (10.4, 1.4), arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.5, connectionstyle="arc3,rad=-0.9"))
    ax.text(10.4, 1.55, "Earth's rotation", ha="center", va="bottom", fontsize=13, color="#5a5a5a")
    return ax


# --- the ML dataset ----------------------------------------------------------------------------------------
def black_box(inputs=("wind (uas, vas)", "ice concentration (sic)", "day of year", "position (lat, lon)"), outputs=("ice velocity (siu, siv)",), model="machine\nlearning\nmodel"):
    """X -> model -> Y, with the tutorial's inputs and outputs."""
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((4.1, 0.7), 1.8, 1.8, boxstyle="round,pad=0.08", fc=INK, ec="none"))
    ax.text(5, 1.6, model, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(1.6, 2.75, "X: inputs", ha="center", fontsize=15, fontweight="bold")
    ax.text(8.4, 2.75, "Y: outputs", ha="center", fontsize=15, fontweight="bold")
    for k, s in enumerate(inputs):
        ax.text(1.6, 2.25 - 0.45 * k, s, ha="center", va="center", fontsize=13)
    for k, s in enumerate(outputs):
        ax.text(8.4, 1.6 - 0.45 * k, s, ha="center", va="center", fontsize=13)
    ax.text(1.6, 0.2, "known everywhere", ha="center", fontsize=12, style="italic", color=GREY)
    ax.text(8.4, 0.2, "known only where there is an instrument", ha="center", fontsize=12, style="italic", color=GREY)
    for x0, x1 in ((3.2, 4.0), (6.0, 6.8)):
        ax.add_patch(FancyArrowPatch((x0, 1.6), (x1, 1.6), arrowstyle="-|>", mutation_scale=22, color=INK, lw=2))
    return ax


def split_diagram(split=None):
    """The train / validation / test split by ensemble member."""
    split = D.SPLIT if split is None else split
    fig, ax = plt.subplots(figsize=(11, 1.9)); ax.set_xlim(0, 31); ax.set_ylim(-1.3, 1.6); ax.axis("off")
    colors = {"train": ORANGE, "val": BLUE, "test": INK}
    for k, m in enumerate(D.MEMBERS):
        part = next(s for s, mem in split.items() if m in mem)
        ax.add_patch(FancyBboxPatch((k + 0.08, 0), 0.84, 1, boxstyle="square,pad=0", fc=colors[part], ec="white"))
        ax.text(k + 0.5, 0.5, m[:-6], ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    for part, lab, x, ha in (("train", "training", 10.5, "center"), ("val", "validation", 21.5, "right"), ("test", "test (r1, r27-r31)", 31, "right")):
        ax.text(x, -0.55, f"{lab}: {len(split[part])} members", ha=ha, fontsize=12, color=colors[part])
    ax.text(15.5, 1.3, f"the 31 ensemble members of IPSL-CM6A-LR, each sampled at the same {D.n_samples:,} ITP positions and dates", ha="center", fontsize=12, color=GREY)
    return ax


def feature_ranges(p):
    """Each input's range before and after scaling (training set)."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    for ax, X, title in zip(axs, (p.X_tr, p.x_tr), ("raw inputs", "scaled inputs")):
        lo, hi = X.min(0), X.max(0)
        ax.barh(D.FEATURES, hi - lo, left=lo, color=LAVENDER)
        for i, f in enumerate(D.FEATURES):
            ax.text(hi[i], i, f"  [{lo[i]:.2f}, {hi[i]:.2f}]", va="center", fontsize=10)
        ax.axvline(0, color=INK, lw=0.8); ax.set_title(title); ax.invert_yaxis()
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].set_xscale("symlog", linthresh=10); axs[0].set_xlabel("value (symlog axis)"); axs[1].set_xlabel("value")
    return axs


# --- the network -------------------------------------------------------------------------------------------
def network_diagram(hidden=(32, 32), show=5, activation="relu"):
    """The architecture used in the tutorial — 7 inputs, hidden layers, 2 outputs — with a few nodes drawn per layer,
    and where the weights (connections), biases (nodes) and activations (node outputs) live."""
    layers = [len(D.FEATURES), *hidden, len(D.TARGETS)]
    names = [D.FEATURES, *[None] * len(hidden), D.TARGETS]
    DX, DY, R = 3.6, 0.8, 0.26
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_aspect("equal"); ax.axis("off")
    pos = []
    for k, n in enumerate(layers):
        shown = n if names[k] is not None else min(n, show)
        ys = (np.arange(shown) - (shown - 1) / 2) * -DY
        pos.append([(k * DX, y) for y in ys])
    # connections: weights
    for a_, b_ in zip(pos[:-1], pos[1:]):
        for (xa, ya) in a_:
            for (xb, yb) in b_:
                ax.plot([xa, xb], [ya, yb], color="#c8c8c8", lw=0.7, zorder=1)
    # one node in the first hidden layer, highlighted: its incoming weights, its bias, its activation
    hx, hy = pos[1][len(pos[1]) // 2]
    for j, (xa, ya) in enumerate(pos[0]):
        ax.plot([xa, hx], [ya, hy], color=PURPLE, lw=1.8, zorder=2)
        xm, ym = xa + 0.42 * (hx - xa), ya + 0.42 * (hy - ya)
        ax.text(xm, ym + 0.1, f"$w_{j + 1}$", color=PURPLE, fontsize=12, ha="center", va="bottom", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
    bx, by = hx - 0.75, hy - 0.75                                           # the bias enters the node from the side
    ax.add_patch(Circle((bx, by), 0.17, fc="white", ec=PURPLE, lw=1.8, zorder=4))
    ax.text(bx, by, "$b$", color=PURPLE, fontsize=13, ha="center", va="center", zorder=5)
    d = np.hypot(hx - bx, hy - by)
    ax.annotate("", (hx - R * (hx - bx) / d, hy - R * (hy - by) / d), (bx + 0.17 * (hx - bx) / d, by + 0.17 * (hy - by) / d),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8), zorder=4)
    top = pos[0][0][1]
    ax.annotate("$a = f\\left(\\sum_i w_i x_i + b\\right)$", (hx + R * 0.7, hy + R * 0.7), (1.5 * DX, top + 1.55), fontsize=15, color=PURPLE,
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.2, shrinkB=4), ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PURPLE, lw=1.2))
    # nodes
    for k, layer in enumerate(pos):
        for j, (x, y) in enumerate(layer):
            is_in, is_out = names[k] is not None and k == 0, names[k] is not None and k > 0
            fc = LAVENDER if is_in else ("#f3f0fa" if is_out else "white")
            ec = PURPLE if (k == 1 and j == len(layer) // 2) else INK
            ax.add_patch(Circle((x, y), R, fc=fc, ec=ec, lw=2.2 if ec == PURPLE else 1.3, zorder=3))
            if not is_in:
                ax.text(x, y, "$f$", ha="center", va="center", fontsize=14, zorder=4)
        if names[k] is not None:
            for (x, y), nm in zip(layer, names[k]):
                ax.text(x - R - 0.15 if k == 0 else x + R + 0.15, y, nm, ha="right" if k == 0 else "left", va="center", fontsize=13)
        elif layers[k] > len(layer):
            ax.text(layer[-1][0], layer[-1][1] - R - 0.15, "⋮", ha="center", va="top", fontsize=16, color=GREY)
            ax.text(layer[-1][0], layer[-1][1] - R - 0.65, f"{layers[k]} nodes", ha="center", va="top", fontsize=12, color=GREY)
        title = ["inputs", *[f"hidden layer {i + 1}" for i in range(len(hidden))], "outputs"][k]
        ax.text(layer[0][0], pos[0][0][1] + R + 0.35, title, ha="center", va="bottom", fontsize=14, fontweight="bold")
    n_par = sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))
    ax.text(DX * (len(layers) - 1) / 2, pos[0][-1][1] - 1.25,
            f"weights $w$ on every connection  ·  one bias $b$ per node  ·  activation $f$ = {activation} inside every node  ·  {n_par:,} parameters",
            ha="center", va="top", fontsize=12.5, color=INK)
    ax.set_xlim(-1.6, DX * (len(layers) - 1) + 1.6); ax.set_ylim(pos[0][-1][1] - 1.8, pos[0][0][1] + 2.1)
    return ax


def activations():
    """The activation functions — sigmoid, tanh, ReLU — and what they do to a node: off, partly on, on."""
    z = np.linspace(-6, 6, 300)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    funcs = (("sigmoid   $\\sigma(z) = 1 / (1 + e^{-z})$", 1 / (1 + np.exp(-z)), "node partly on / off"),
             ("tanh$(z)$", np.tanh(z), "node partly on / off"),
             ("ReLU   $\\max(0, z)$", np.maximum(0, z), None))
    for ax, (name, f, mid) in zip(axs, funcs):
        ax.plot(z, f, color=PURPLE, lw=2.5); ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
        ax.set_title(name, fontsize=14); ax.set_xlabel("$z = \\sum_i w_i x_i + b$")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.25 * (hi - lo), hi)
        ax.text(-5.8, lo - 0.12 * (hi - lo), "node OFF", color="#b03a2e", fontsize=12, fontweight="bold", ha="left", va="center")
        ax.text(5.8, lo - 0.12 * (hi - lo), "node ON", color="#2e7d32", fontsize=12, fontweight="bold", ha="right", va="center")
        if mid:
            ax.text(0, lo - 0.12 * (hi - lo), mid, color="#1f78b4", fontsize=12, fontweight="bold", ha="center", va="center")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return axs


def residual_histogram(Y_true, Y_pred, Y_lin=None):
    """Distribution of the vector error for the network (and the linear model)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    e = np.hypot(*(Y_pred - Y_true).T)
    ax.hist(e, bins=80, range=(0, 0.15), color=LAVENDER, label=f"network  (RMS {np.sqrt(np.mean(e ** 2)):.3f} m/s)")
    if Y_lin is not None:
        el = np.hypot(*(Y_lin - Y_true).T)
        ax.hist(el, bins=80, range=(0, 0.15), histtype="step", color=INK, label=f"linear  (RMS {np.sqrt(np.mean(el ** 2)):.3f} m/s)")
    ax.set_xlabel("|predicted − true| [m/s]"); ax.set_ylabel("samples"); ax.legend(frameon=False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ax
