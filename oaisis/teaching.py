"""Figures that explain the machine-learning concepts with the tutorial's own data
(the ice velocity / wind relation along the ITP tracks)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

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
            ax.text(0.98, 0.04, "error $e_i$ = model − truth", transform=ax.transAxes, ha="right", fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
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
    ax.annotate(f"minimum: w = {w_opt:.3f}", (w_opt, mse(xs, ys, w_opt, b)), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=10)
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
    ax.set_title(title if title is not None else f"learning rate {lr:g}: {steps} steps, final w = {ws[-1]:.3f}", fontsize=10)
    return ws


def gradient_descent_panel(xs, ys, lrs=(0.1, 0.8, 2.1), steps=15):
    """Too small, about right, too large: the same descent with three learning rates."""
    fig, axs = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 4.2), layout="constrained")
    for ax, lr, lab in zip(axs, lrs, ("too small", "about right", "too large")):
        gradient_descent(xs, ys, lr=lr, steps=steps, ax=ax, title=f"learning rate {lr:g} — {lab}")
    return axs


# --- the ML dataset ----------------------------------------------------------------------------------------
def black_box(inputs=("wind (uas, vas)", "ice concentration (sic)", "day of year", "position (lat, lon)"), outputs=("ice velocity (siu, siv)",), model="machine\nlearning\nmodel"):
    """X -> model -> Y, with the tutorial's inputs and outputs."""
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((4.1, 0.7), 1.8, 1.8, boxstyle="round,pad=0.08", fc=INK, ec="none"))
    ax.text(5, 1.6, model, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(1.6, 2.75, "X: inputs", ha="center", fontsize=15, fontweight="bold")
    ax.text(8.4, 2.75, "Y: outputs", ha="center", fontsize=15, fontweight="bold")
    for k, s in enumerate(inputs):
        ax.text(1.6, 2.25 - 0.45 * k, s, ha="center", va="center", fontsize=11)
    for k, s in enumerate(outputs):
        ax.text(8.4, 1.6 - 0.45 * k, s, ha="center", va="center", fontsize=11)
    ax.text(1.6, 0.2, "known everywhere", ha="center", fontsize=10, style="italic", color=GREY)
    ax.text(8.4, 0.2, "known only where there is an instrument", ha="center", fontsize=10, style="italic", color=GREY)
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
        ax.text(k + 0.5, 0.5, m[:-6], ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for part, lab, x, ha in (("train", "training", 10.5, "center"), ("val", "validation", 21.5, "right"), ("test", "test (r1, r27-r31)", 31, "right")):
        ax.text(x, -0.55, f"{lab}: {len(split[part])} members", ha=ha, fontsize=10, color=colors[part])
    ax.text(15.5, 1.3, f"the 31 ensemble members of IPSL-CM6A-LR, each sampled at the same {D.n_samples:,} ITP positions and dates", ha="center", fontsize=10, color=GREY)
    return ax


def feature_ranges(p):
    """Each input's range before and after scaling (training set)."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    for ax, X, title in zip(axs, (p.X_tr, p.x_tr), ("raw inputs", "scaled inputs")):
        lo, hi = X.min(0), X.max(0)
        ax.barh(D.FEATURES, hi - lo, left=lo, color=LAVENDER)
        for i, f in enumerate(D.FEATURES):
            ax.text(hi[i], i, f"  [{lo[i]:.2f}, {hi[i]:.2f}]", va="center", fontsize=8)
        ax.axvline(0, color=INK, lw=0.8); ax.set_title(title); ax.invert_yaxis()
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axs[0].set_xscale("symlog", linthresh=10); axs[0].set_xlabel("value (symlog axis)"); axs[1].set_xlabel("value")
    return axs


# --- the network -------------------------------------------------------------------------------------------
def network_diagram(hidden=(32, 32), show=5, activation="relu"):
    """The architecture used in the tutorial: 7 inputs, hidden layers, 2 outputs — drawn with a few nodes per layer."""
    layers = [len(D.FEATURES), *hidden, len(D.TARGETS)]
    names = [D.FEATURES, *[None] * len(hidden), D.TARGETS]
    fig, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(-0.8, len(layers) - 0.2); ax.set_ylim(-3.6, 3.8); ax.axis("off")
    pos = []
    for k, n in enumerate(layers):
        shown = min(n, show if names[k] is None else n)
        ys = np.linspace(1.0, -1.0, shown) * (0.45 * (shown - 1)) if shown > 1 else np.array([0.0])
        pos.append([(k, y) for y in ys])
        for j, y in enumerate(ys):
            ax.add_patch(Circle((k, y), 0.17, fc="white" if names[k] is None else LAVENDER, ec=INK, lw=1.2, zorder=3))
            if names[k] is None:
                ax.text(k, y, "f", ha="center", va="center", fontsize=10, style="italic", zorder=4)
        if names[k] is not None:
            for y, nm in zip(ys, names[k]):
                ax.text(k - 0.28 if k == 0 else k + 0.28, y, nm, ha="right" if k == 0 else "left", va="center", fontsize=10)
        if names[k] is None and n > shown:
            ax.text(k, ys[-1] - 0.5, f"⋮\n{n} nodes", ha="center", va="top", fontsize=10, color=GREY)
        label = ["input layer", *[f"hidden layer {i + 1}\n({activation})" for i in range(len(hidden))], "output layer"][k]
        ax.text(k, 2.9, label, ha="center", fontsize=10, fontweight="bold")
    for a, b in zip(pos[:-1], pos[1:]):
        for (xa, ya) in a:
            for (xb, yb) in b:
                ax.plot([xa, xb], [ya, yb], color=GREY, lw=0.4, zorder=1)
    n_par = sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))
    ax.text((len(layers) - 1) / 2, -3.3, f"every line is a weight w, every node adds a bias b and applies f:  {n_par:,} parameters in total", ha="center", fontsize=10, color=INK)
    return ax


def activations():
    """The activation functions: sigmoid, tanh, ReLU."""
    z = np.linspace(-6, 6, 300)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
    for ax, (name, f) in zip(axs, (("sigmoid  σ(z) = 1 / (1 + e⁻ᶻ)", 1 / (1 + np.exp(-z))), ("tanh(z)", np.tanh(z)), ("ReLU  max(0, z)", np.maximum(0, z)))):
        ax.plot(z, f, color=PURPLE, lw=2); ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
        ax.set_title(name, fontsize=11); ax.set_xlabel("z = Σ wᵢxᵢ + b")
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
