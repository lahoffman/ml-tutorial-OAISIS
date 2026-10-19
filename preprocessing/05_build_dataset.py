"""Combine the outputs of 03 and 04 into data/oaisis_tutorial_IPSL.nc.

    python preprocessing/05_build_dataset.py

This is the end of the from-scratch pipeline (00 -> 05). The shipped dataset was instead produced by
05b_build_dataset_from_zenodo.py from the 2025 intermediate files; both write the same layout.
"""
import xarray as xr

from config import DATASET, INTERMEDIATE
from dataset import write_dataset


def main():
    obs = xr.open_dataset(INTERMEDIATE / "colocated_ipsl.nc")
    maps = xr.open_dataset(INTERMEDIATE / "maps_ipsl.nc")
    ds = write_dataset(obs, maps, DATASET, "built from scratch with preprocessing/00-05")
    print(ds)


if __name__ == "__main__":
    main()
