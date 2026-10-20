"""Observing system experiments: retrain the network on subsets of the ITP samples and score the reconstruction.

    results = run_experiment("A", {"ITP 95": tracks(95), "ITP 5": tracks(5)})
    steps   = run_sequence("B", [1, 2, 3, 4, 96, 24, 7, 27, 5, 15])
"""
import time

import numpy as np

from . import data as D
from . import plots
from .metrics import cell_scores, skill
from .ml import HP, train_model
from .selectors import describe, everything, tracks

SEEDS = (0, 1, 2)                       # networks per observing system (the spread shows the training noise)
HP_OSD = dict(HP, epochs=15)            # the training curve is flat by then; keeps the loops short
_cache = {}


def train_on(sel, name, seeds=SEEDS, hp=HP_OSD):
    """Retrain on a selection of ITP samples (once per seed); pooled skills and the seed-mean skill map.
    Results are cached, so re-running the same selection with the same settings returns instantly."""
    key = (sel.tobytes(), tuple(seeds), tuple(D.FEATURES), tuple(sorted((k, str(v)) for k, v in hp.items())))
    if key in _cache:
        res = dict(_cache[key], name=name)
        print(f"{name:14s} {describe(sel)}   skill {res['skills'].mean():.3f}  (cached)")
        return res
    t0 = time.time()
    p = D.prepared(sel)
    skills, maps = [], []
    for seed in seeds:
        mdl, _ = train_model(p.x_tr, p.y_tr, p.x_va, p.y_va, hp=hp, seed=seed)
        X = D.maps_X(p.features)
        pred = np.full((X.shape[0], 2), np.nan, dtype="float32")
        pred[D.maps.ok] = p.scaler.y_inv(mdl.predict(p.scaler.x(X[D.maps.ok]), verbose=0))
        Y_rec = pred.reshape(D.maps.ny, D.maps.nx, D.maps.nt, 2)
        skills.append(skill(D.maps.Y_true[D.maps.ice_mask].reshape(-1, 2), Y_rec[D.maps.ice_mask].reshape(-1, 2)))
        maps.append(cell_scores(Y_rec)[0])
    res = dict(name=name, sel=sel, n_obs=int(sel.sum()), skills=np.array(skills), skill_map=np.nanmean(maps, axis=0))
    _cache[key] = res
    print(f"{name:14s} {describe(sel)}   skill {res['skills'].mean():.3f}  (seeds: {' '.join(f'{v:.3f}' for v in skills)})   "
          f"{len(seeds)} network{'s' if len(seeds) > 1 else ''} x {hp['epochs']} epochs in {time.time() - t0:.0f} s")
    return res


def _settings(quick, hp):
    """Seeds and hyperparameters for a run: the notebook's HP (or ml.HP) with 15 epochs; quick = one seed, 10 epochs."""
    base = dict(HP if hp is None else hp)
    return ((0,), dict(base, epochs=10)) if quick else (SEEDS, dict(base, epochs=min(15, base["epochs"])))


def run_experiment(name, scenarios, quick=False, maps=True, hp=None):
    """Train and score every observing system in `scenarios` (plus the `all` reference);
    draw the sampling maps, the skill bars and the skill-difference maps. `hp`: the hyperparameters to use (default ml.HP)."""
    seeds, hp = _settings(quick, hp)
    scenarios = {**scenarios, "all": everything()}
    print(f"{name}: {len(scenarios)} observing systems x {len(seeds)} seed(s)\n")
    results = {k: train_on(v, k, seeds, hp) for k, v in scenarios.items()}
    plots.sampling(scenarios, title=name)
    plots.skill_bars(results, title=name)
    if maps:
        plots.skill_difference(results)
    return results


def run_sequence(name, sequence, quick=False, labels=None, map_steps=None, hp=None):
    """Observing systems that grow one ITP at a time: skill curve against the `all` reference, plus maps at `map_steps`."""
    seeds, hp = _settings(quick, hp)
    print(f"{name}: {len(sequence)} steps x {len(seeds)} seed(s); order of addition {list(sequence)}\n")
    steps = [train_on(tracks(*sequence[:k]), f"{k} ITP{'s' if k > 1 else ''}", seeds, hp) for k in range(1, len(sequence) + 1)]
    ref = train_on(everything(), "all", seeds, hp)
    plots.skill_curve(steps, ref, title=name, labels=labels or {})
    if map_steps:
        plots.sampling({steps[k - 1]["name"]: steps[k - 1]["sel"] for k in map_steps}, title=f"{name}: observing systems at the mapped steps")
        plots.skill_difference({"all": ref, **{steps[k - 1]["name"]: steps[k - 1] for k in map_steps}})
    return steps


# --- the order of addition used in experiment B -------------------------------------------------------
def centroid(ids):
    m = tracks(*ids); la, lo = np.radians(D.lat_obs[m]), np.radians(D.lon_obs[m])
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]).mean(1)


def mean_lon(i):
    lo = np.radians(D.lon_obs[D.track_obs == i]); return np.degrees(np.arctan2(np.sin(lo).mean(), np.cos(lo).mean()))


def farthest_first_sequence(n_beaufort=4, n_elsewhere=4, extra=(5, 15), min_days=100):
    """Start with the long Beaufort Sea tracks, then always add the long track farthest from everything observed so far."""
    big = D.track_ids[D.track_len >= min_days]
    beaufort = [int(i) for i in big if mean_lon(i) < -120][:n_beaufort]
    candidates = sorted([int(i) for i in big if mean_lon(i) > -90], key=lambda i: -D.track_len[D.track_ids == i][0])[:n_elsewhere]
    elsewhere = []
    while candidates:
        have = centroid(beaufort + elsewhere)
        far = max(candidates, key=lambda i: np.linalg.norm(centroid([i]) - have))
        elsewhere.append(far); candidates.remove(far)
    return beaufort + elsewhere + list(extra)
