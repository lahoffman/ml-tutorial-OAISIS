"""Makes `import oaisis` work from anywhere: Binder runs `pip install .` because this file exists
(the metadata is in pyproject.toml). `reconstruction.ipynb` does not need it; the Jupyter Book pages do."""
from setuptools import setup

setup()
