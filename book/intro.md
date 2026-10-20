# Reconstructing Arctic sea-ice velocity with a neural network

*An Observing System Design tutorial — OAISIS*

Some quantities are known everywhere: winds from reanalysis, ice concentration from satellites, the date, the map.
Others are only measured where an instrument happens to be — sea-ice velocity from a drifting buoy, say. If we can
learn how the first group relates to the second *at the instrument locations*, we can predict the second everywhere.
How well that works depends on where the instruments were — and that dependence is what **Observing System Design**
studies.

This book does it in six short steps with a small neural network, climate-model output standing in for the
measurements, and the tracks of real Arctic drifting instruments. By the end you will train networks on observing
systems of your own design and see where their reconstructions succeed and fail.

## The idea in two pictures

**Training.** A neural network learns the relation *inputs → ice velocity* at the observation locations. Each observing
system gives a different trained network: ① a single Ice-Tethered Profiler (ITP 95), ② the same ITP plus a second one
elsewhere (ITP 5).

```{figure} ../figures/training.png
:width: 100%
:name: training
```

**Reconstruction.** Each trained network is then given the inputs on the *full grid* and predicts ice velocity
everywhere. Because the "world" is a climate model, the true velocity is known everywhere too, so the prediction can be
scored honestly (①, ②) — and the two scores compared (③): did the added observations improve the reconstruction?

```{figure} ../figures/reconstruction.png
:width: 100%
:name: reconstruction
```

Both networks are real: they are the `ITP 95` and `ITP 95 + 5` observing systems of [experiment A](06_osd.ipynb) on the last page.

## Contents

| | page | what you do |
|---|---|---|
| 1 | [The data](01_data.ipynb) | meet the ITPs and the climate-model testbed; maps of the inputs |
| 2 | [Building the ML dataset](02_ml_dataset.ipynb) | choose inputs and outputs; split; scale |
| 3 | [Linear regression](03_linear_regression.ipynb) | the simplest model, on our data: weights, bias, loss |
| 4 | [The neural network](04_neural_network.ipynb) | architecture, gradient descent, learning curves, overfitting, test skill |
| 5 | [Reconstruction](05_reconstruction.ipynb) | from ITP tracks to the whole Arctic |
| 6 | [Observing System Design](06_osd.ipynb) | experiments A and B, then design your own |

## How to use this book

Every page is a notebook, and the code on each page is what you see — the helper functions it calls live in the
small [`oaisis`](https://github.com/lahoffman/ml-tutorial-OAISIS/tree/main/oaisis) package so that the page shows only
what matters for the concept at hand.

* **Just read.** All outputs are pre-computed.
* **Live Code** (rocket icon at the top of a page → *Live Code*). A Python kernel starts for you on
  [mybinder.org](https://mybinder.org) — a minute or two normally, longer if nobody has visited since the last update —
  and every code cell on the page gets a *Run* button and becomes editable. **First click *Restart & Run All*** (it
  appears next to the status line once the kernel is ready): the cells share one kernel, so later cells need the
  imports and variables of earlier ones. After that, change a number, re-run that cell, see what happens. The kernel is
  yours alone: nothing you do affects other participants or the book itself, and if you break something, **reload the
  page** to get the original code back (your edits are only in your browser).
* **Binder** (rocket icon → *Binder*) opens the page as a full notebook in JupyterLab, again on a server of your own,
  with the whole repository alongside. Nothing is saved when the server stops — download the notebook if you want to keep it.
* **Locally.** `git clone https://github.com/lahoffman/ml-tutorial-OAISIS`, `pip install -r requirements.txt -e .`,
  then open any page under `book/` in Jupyter.

The whole tutorial also exists as **one self-contained notebook**,
[`reconstruction.ipynb`](https://github.com/lahoffman/ml-tutorial-OAISIS/blob/main/reconstruction.ipynb), with every
line of code written out — the place to look when you want to see *how* something is done rather than *what* it shows,
and the version used in live sessions: open it in
[Colab](https://colab.research.google.com/github/lahoffman/ml-tutorial-OAISIS/blob/main/reconstruction.ipynb) or
[Binder](https://mybinder.org/v2/gh/lahoffman/ml-tutorial-OAISIS/HEAD?labpath=reconstruction.ipynb).

## Credits

ITP positions: Ice-Tethered Profiler Program, Woods Hole Oceanographic Institution ([whoi.edu/itp](https://www.whoi.edu/itp);
Toole et al. 2011; Krishfield et al. 2008). Model output: IPSL-CM6A-LR (Boucher et al. 2020) via CMIP6 / ESGF.
Tutorial: Lauren Hoffman (UC Santa Cruz).
