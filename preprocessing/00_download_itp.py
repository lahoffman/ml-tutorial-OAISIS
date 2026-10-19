"""Download the WHOI Ice-Tethered Profiler 'final' data products.

Not run for the shipped dataset (the ITP positions were taken from files already on disk);
kept so the pipeline is reproducible from scratch.

    python preprocessing/00_download_itp.py

Source: https://scienceweb.whoi.edu/itp/data/  (Toole et al. 2011; Krishfield et al. 2008)
"""
import urllib.request
import zipfile

from config import ITP_NUMBERS, ITP_URL, RAW


def main():
    out = RAW / "itp"
    out.mkdir(parents=True, exist_ok=True)
    for n in ITP_NUMBERS:
        zpath = out / f"itp{n}final.zip"
        if (out / f"itp{n}final.mat").exists():
            continue
        try:
            urllib.request.urlretrieve(ITP_URL.format(n=n), zpath)
        except Exception as e:                      # some numbers have no 'final' product
            print(f"itp{n}: skipped ({e})")
            continue
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        zpath.unlink()
        print(f"itp{n}: ok")


if __name__ == "__main__":
    main()
