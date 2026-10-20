# ML tutorial: Observing System Design for Arctic sea-ice velocity

*OAISIS — AI-based Observing System Design for polar sea ice*

A hands-on tutorial that trains a small neural network to predict sea-ice velocity from wind,
ice concentration and position, using CMIP6 model output sampled along real Ice-Tethered Profiler
(ITP) tracks as **synthetic observations**. The trained network reconstructs ice velocity over the
whole Arctic, and retraining it with different subsets of the synthetic observations shows where
additional observations would matter most — the Observing System Design question.

## Run it

| | |
|---|---|
| Colab | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lahoffman/ml-tutorial-OAISIS/blob/main/reconstruction.ipynb) — the notebook; starts in seconds (Google account needed; the first cell fetches the data) |
| Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lahoffman/ml-tutorial-OAISIS/HEAD?labpath=reconstruction.ipynb) — the notebook, no account needed; takes a minute or two to start ([mirror at GESIS](https://notebooks.gesis.org/binder/v2/gh/lahoffman/ml-tutorial-OAISIS/HEAD?labpath=reconstruction.ipynb) if mybinder.org is busy) |
| Locally | `pip install -r requirements.txt jupyterlab && jupyter lab reconstruction.ipynb` (Python 3.10–3.12) |
| Website | **[lahoffman.github.io/ml-tutorial-OAISIS](https://lahoffman.github.io/ml-tutorial-OAISIS)** — the tutorial as a Jupyter Book, one concept per page, for reading and self-study (with *Live Code* on a Binder kernel) |

Everything the notebook needs is in this repository (`data/oaisis_tutorial_IPSL.nc`, 7 MB). Sections 1–3 run in
a couple of minutes on a laptop CPU; the observing system experiments in section 4 retrain the network ~45 times
and take 10–15 minutes (set `SEEDS = (0,)` or pass `quick=True` for a faster look).

## What's here

```
reconstruction.ipynb     the tutorial as one self-contained notebook: every line of code in one place
book/                    the same tutorial as a Jupyter Book (the website): one page per concept, barebones code
oaisis/                  the helper package the book pages import (data, plots, ml, metrics, selectors, experiments, teaching);
                         `pip install -e .` -- Binder does this automatically
data/                    the dataset and its description
preprocessing/           how the dataset was made: download -> ITP positions -> co-location -> maps -> dataset
figures/                 the three schematics and the script that draws them (it trains the small networks they show)
requirements.txt         python packages (also used by Binder)
.github/workflows/       builds the book and publishes it to GitHub Pages on every push to main
```

## The website

`jupyter-book build book` builds the site into `book/_build/html` (`pip install "jupyter-book==1.0.*"` first; the first
build executes every page and takes 15-20 minutes because the observing system page retrains the network ~50 times —
later builds re-run only the pages whose code changed). The GitHub Action does the same on every push and deploys to
GitHub Pages; the one-time setup is *Settings → Pages → Source: GitHub Actions* in the repository.

**Live Code** on the website is [Thebe](https://thebe.readthedocs.io) backed by mybinder.org: each visitor gets a
kernel of their own, built from this repository's `requirements.txt`, and edits live only in their browser — reloading a
page restores it. Cells share one kernel, so use *Restart & Run All* once before editing individual cells.

## Running a live session

Binder rebuilds its image after any push that touches `requirements.txt`, `setup.py` or `oaisis/` (10–15 min), so open the
Binder link once the day before. mybinder.org is a free service with limited capacity: ask participants to click Colab or
Binder at the very start, while the slides run, and point those without a Google account to Binder (GESIS mirror as
fallback). Participants who installed locally beforehand (`requirements.txt`, Python 3.10–3.12) need none of this.

## Data

`data/oaisis_tutorial_IPSL.nc` holds two things from the IPSL-CM6A-LR CMIP6 historical simulation:
31 ensemble members sampled at 11 242 daily ITP positions (2004–2014), and ten days of Arctic maps from
one member (Nov–Dec 2000). Variables, units and provenance are in [`data/README.md`](data/README.md);
the scripts that produced the file are in [`preprocessing/`](preprocessing/README.md).

## Credits

ITP positions: Ice-Tethered Profiler Program, Woods Hole Oceanographic Institution
([whoi.edu/itp](https://www.whoi.edu/itp); Toole et al. 2011; Krishfield et al. 2008).
Model output: IPSL-CM6A-LR (Boucher et al. 2020) via CMIP6 / ESGF (Eyring et al. 2016).
Tutorial: Lauren Hoffman (UC Santa Cruz).
