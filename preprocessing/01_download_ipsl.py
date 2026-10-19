"""Download IPSL-CM6A-LR historical daily output from ESGF.

Not run for the shipped dataset (files were already on the cluster); kept for reproducibility.
Five variables x 31 members x 165 years of daily data is a few hundred GB — run on a machine
with the space, or restrict MEMBERS in config.py.

    python preprocessing/01_download_ipsl.py            # everything in config.MEMBERS
    python preprocessing/01_download_ipsl.py r1i1p1f1   # one member

Variables: siu, siv (sea-ice velocity, grid x/y components!), siconc, uas, vas.
"""
import sys
import urllib.request

from config import IPSL_FILES, MEMBERS, RAW, ipsl_url


def main(members):
    for member in members:
        for var in IPSL_FILES:
            url, name = ipsl_url(var, member)
            out = RAW / "ipsl" / var
            out.mkdir(parents=True, exist_ok=True)
            if (out / name).exists():
                continue
            print("downloading", name)
            urllib.request.urlretrieve(url, out / name)


if __name__ == "__main__":
    main(sys.argv[1:] or MEMBERS)
