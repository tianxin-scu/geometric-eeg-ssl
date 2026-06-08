"""Build signal caches (9 subjects/dataset) on the local machine.

Mirrors `notebooks/colab_pretrain.ipynb` section 4b for local execution.
Downloads raw data via MNE/MOABB if not present, then writes the signal
cache into EEG_CACHE_DIR.

Env vars (set before running, or override CLI args):
    EEG_CACHE_DIR   where signal caches go      (default: ./cache)
    MNE_DATA        where MNE downloads raw EEG (default: ./mne_data)

Usage:
    python scripts/build_caches.py
    python scripts/build_caches.py --n-subjects 9 --cache-dir D:/eeg_cache
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=9,
                   help="Subjects per dataset (default 9; matches Option A).")
    p.add_argument("--datasets", default="physionet_mi,bcic_2b,sleep_edfx",
                   help="Comma-separated subset to build (default: all three). "
                        "Useful to build one dataset at a time, e.g. when "
                        "sleep_edfx raw data arrives via SCP separately.")
    p.add_argument("--cache-dir", default=None,
                   help="Override EEG_CACHE_DIR (default: ./cache).")
    p.add_argument("--mne-dir", default=None,
                   help="Override MNE_DATA (default: ./mne_data).")
    args = p.parse_args()
    wanted = {d.strip() for d in args.datasets.split(",")}

    cache_dir = Path(args.cache_dir or os.environ.get(
        "EEG_CACHE_DIR", _repo_root / "cache")).resolve()
    mne_dir = Path(args.mne_dir or os.environ.get(
        "MNE_DATA", _repo_root / "mne_data")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    mne_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EEG_CACHE_DIR"] = str(cache_dir)
    os.environ["MNE_DATA"] = str(mne_dir)

    print(f"EEG_CACHE_DIR -> {cache_dir}")
    print(f"MNE_DATA      -> {mne_dir}")

    # MNE's load_data helpers prompt interactively to remember the dataset
    # path the first time they run. Set the relevant config keys upfront
    # so they don't ask -- this script is non-interactive.
    import mne
    mne.set_config("MNE_DATA", str(mne_dir))
    mne.set_config("MNE_DATASETS_EEGBCI_PATH", str(mne_dir))
    mne.set_config("PHYSIONET_SLEEP_PATH", str(mne_dir))

    from src.config import Config
    from src.datasets.bcic_2b import ALL_SUBJECTS as BCIC_ALL
    from src.datasets.bcic_2b import BCIC2B
    from src.datasets.physionet_mi import EXCLUDED_SUBJECTS, PhysioNetMI
    from src.datasets.sleep_edfx import ALL_SUBJECTS as SLEEP_ALL
    from src.datasets.sleep_edfx import SleepEDFx

    N = args.n_subjects
    cfg = Config()

    phys_subj = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS][:N]
    bcic_subj = list(BCIC_ALL)[:N]
    sleep_subj = list(SLEEP_ALL)[:N]

    if "physionet_mi" in wanted:
        print(f"\n--- physionet_mi ({len(phys_subj)} subjects) ---")
        PhysioNetMI(subjects=phys_subj, cfg=cfg, mode="pretrain", verbose=True).load()
    if "bcic_2b" in wanted:
        print(f"\n--- bcic_2b ({len(bcic_subj)} subjects) ---")
        BCIC2B(subjects=bcic_subj, cfg=cfg, mode="pretrain", verbose=True).load()
    if "sleep_edfx" in wanted:
        print(f"\n--- sleep_edfx ({len(sleep_subj)} subjects) ---")
        SleepEDFx(subjects=sleep_subj, cfg=cfg, mode="pretrain", verbose=True).load()

    print("\ndone. cache contents:")
    for f in sorted(cache_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
