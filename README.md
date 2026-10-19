# ML tutorial: observing-system design for Arctic sea-ice velocity

*OAISIS — AI-based observing system design for polar sea ice (Antarctica InSync / AI4InSync)*

A hands-on tutorial that trains a small neural network to predict sea-ice velocity from wind,
ice concentration and position, using CMIP6 model output sampled along real Ice-Tethered Profiler
(ITP) tracks as **synthetic observations**. The trained network reconstructs ice velocity over the
whole Arctic, and retraining it with different subsets of the synthetic observations shows where
additional observations would matter most — the observing-system-design question.

## Run it

| | |
|---|---|
| Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lahoffman/ml-tutorial-OAISIS/HEAD?labpath=reconstruction.ipynb) |
| Locally | `pip install -r requirements.txt jupyterlab && jupyter lab reconstruction.ipynb` (Python 3.10–3.12) |

Everything the notebook needs is in this repository (`data/oaisis_tutorial_IPSL.nc`, 7 MB). It runs in
about five minutes on a laptop CPU.

## What's here

```
reconstruction.ipynb     the tutorial (self-contained)
data/                    the dataset and its description
preprocessing/           how the dataset was made: download -> ITP positions -> co-location -> maps -> dataset
requirements.txt         python packages (also used by Binder)
```

## Data

`data/oaisis_tutorial_IPSL.nc` holds two things from the IPSL-CM6A-LR CMIP6 historical simulation:
31 ensemble members sampled at 11 242 daily ITP positions (2004–2014), and ten days of Arctic maps from
one member (Nov–Dec 2000). Variables, units and provenance are in [`data/README.md`](data/README.md);
the scripts that produced the file are in [`preprocessing/`](preprocessing/README.md).

## Credits

ITP positions: Ice-Tethered Profiler Program, Woods Hole Oceanographic Institution
([whoi.edu/itp](https://www.whoi.edu/itp); Toole et al. 2011; Krishfield et al. 2008).
Model output: IPSL-CM6A-LR (Boucher et al. 2020) via CMIP6 / ESGF (Eyring et al. 2016).
Tutorial: Lauren Hoffman (UC Santa Cruz), first given at the KITP 2025 AI4InSync sessions.
