"""Paths, members and settings shared by the preprocessing scripts.

Nothing in the tutorial reads this file; it documents how data/oaisis_tutorial_IPSL.nc was made.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"                  # downloads (git-ignored)
INTERMEDIATE = ROOT / "data" / "intermediate"  # outputs of scripts 02-04 (git-ignored)
DATASET = ROOT / "data" / "oaisis_tutorial_IPSL.nc"

# --- ITP ---------------------------------------------------------------------------------------
ITP_URL = "https://scienceweb.whoi.edu/itp/data/itp{n}final.zip"
ITP_NUMBERS = [n for n in range(1, 133) if n not in (66, 71)]   # systems with a 'final' product
ITP_END = "2014-12-31"                                            # last day of the CMIP6 historical runs

# --- IPSL-CM6A-LR historical --------------------------------------------------------------------
MODEL = "IPSL-CM6A-LR"
MEMBERS = [f"r{i}i1p1f1" for i in range(1, 32)]                 # 31 members co-located to the ITPs
MAP_MEMBER = "r1i1p1f1"                                          # member used for the Arctic maps
MAP_START, MAP_END = "2000-11-25", "2000-12-04"                  # window stored in the shipped dataset

ESGF = "https://esgf-node.ornl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical"
# table, grid label, data version and file time span per variable
IPSL_FILES = {
    "siu":    dict(table="SIday", grid="gn", version="v20180803"),
    "siv":    dict(table="SIday", grid="gn", version="v20180803"),
    "siconc": dict(table="SIday", grid="gn", version="v20180803"),
    "uas":    dict(table="day",   grid="gr", version="v20180803"),
    "vas":    dict(table="day",   grid="gr", version="v20180803"),
}


def ipsl_span(var, member):
    """File time span: atmosphere files for r12-r31 start in 1950, everything else in 1850."""
    atmos = IPSL_FILES[var]["table"] == "day"
    late = int(member[1:].split("i")[0]) >= 12
    return "19500101-20141231" if (atmos and late) else "18500101-20141231"


def ipsl_url(var, member):
    f = IPSL_FILES[var]
    name = f"{var}_{f['table']}_{MODEL}_historical_{member}_{f['grid']}_{ipsl_span(var, member)}.nc"
    return f"{ESGF}/{member}/{f['table']}/{var}/{f['grid']}/{f['version']}/{name}", name


# --- processing choices --------------------------------------------------------------------------
ARCTIC_LAT_MIN = 55.0        # maps: keep grid cells north of this
FILL_THRESHOLD = 1e19        # CMIP6 fill value is 1e20
