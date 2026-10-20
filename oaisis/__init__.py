"""oaisis — helpers for the OAISIS sea-ice observing-system-design tutorial.

The Jupyter Book pages import from here so that they show only the code that matters for
the concept on the page. `reconstruction.ipynb` is the same code, written out in full in
one self-contained notebook.

    from oaisis import data, plots, ml, selectors, experiments, teaching
"""
import os as _os
import warnings as _w

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # keep TensorFlow's start-up chatter out of the pages

_w.filterwarnings("ignore", category=RuntimeWarning)   # nanmean over all-NaN land cells is expected

from . import data, plots, metrics, ml, selectors, experiments, teaching  # noqa: F401

__all__ = ["data", "plots", "metrics", "ml", "selectors", "experiments", "teaching"]
