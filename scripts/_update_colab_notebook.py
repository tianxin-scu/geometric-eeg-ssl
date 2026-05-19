"""One-shot helper to surgically update notebooks/colab_pretrain.ipynb.

Changes:
1. Replace s4_download cell with a resume-aware version that skips already-cached subjects.
2. Insert a new "Keep Colab alive while laptop sleeps" section between sections 2 and 3.
3. Update the intro cell's section list to mention the new section.

Run from repo root: python scripts/_update_colab_notebook.py
Idempotent: re-running detects existing changes and exits cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pretrain.ipynb"


def code_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


NEW_S4_DOWNLOAD = """import os, mne
mne.set_log_level('WARNING')

# PhysioNet MI: 109 subjects, 4 excluded (88, 92, 100, 104)
EXCLUDED = {88, 92, 100, 104}
ALL_SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED]  # 105 subjects
MI_RUNS = [4, 6, 8, 10, 12, 14]

# MNE caches PhysioNet MI at:
#   {MNE_DATA}/MNE-eegbci-data/files/eegmmidb/1.0.0/S{NNN}/S{NNN}R{RR}.edf
EEGBCI_ROOT = os.path.join(MNE_DATA_DIR, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0')

def subject_fully_cached(subj: int, runs: list[int]) -> bool:
    subj_dir = os.path.join(EEGBCI_ROOT, f'S{subj:03d}')
    if not os.path.isdir(subj_dir):
        return False
    for run in runs:
        edf = os.path.join(subj_dir, f'S{subj:03d}R{run:02d}.edf')
        if not os.path.exists(edf) or os.path.getsize(edf) == 0:
            return False
    return True

# Resume-aware download: skip subjects whose EDF files are already on Drive.
# Safe to re-run after a session reconnect — fully-cached subjects skip in O(stat).
cached = [s for s in ALL_SUBJECTS if subject_fully_cached(s, MI_RUNS)]
todo = [s for s in ALL_SUBJECTS if s not in cached]
print(f'Cached: {len(cached)}/{len(ALL_SUBJECTS)} subjects. To download: {len(todo)}.')

for i, subj in enumerate(todo):
    try:
        mne.datasets.eegbci.load_data(subj, MI_RUNS, path=MNE_DATA_DIR, verbose=False)
    except Exception as e:
        print(f'  subject {subj:3d}: download failed ({e}); will retry on next run')
        continue
    if (i + 1) % 5 == 0 or (i + 1) == len(todo):
        print(f'  downloaded {i+1}/{len(todo)} (subject {subj:3d})')

# Final verification
still_missing = [s for s in ALL_SUBJECTS if not subject_fully_cached(s, MI_RUNS)]
if still_missing:
    print(f'WARNING: {len(still_missing)} subjects still incomplete: {still_missing}')
    print('Re-run this cell to resume; partial files are skipped automatically.')
else:
    print(f'All {len(ALL_SUBJECTS)} subjects ready under {EEGBCI_ROOT}')
"""


KEEPALIVE_HEADER = """## 2b. Keep Colab alive while your laptop sleeps

Colab sessions are tied to the browser tab that owns them. If the laptop sleeps
or the network drops, the WebSocket dies and the runtime can be reclaimed even
though GPU compute is still busy. The training loop saves a checkpoint every 10
epochs, so a disconnect costs at most ~10 epochs — but it's better to avoid one
entirely. Three layers of protection, applied together:

**1. Stop the laptop from sleeping.**
- **macOS:** open Terminal and run `caffeinate -dis &` before closing the lid.
  Or System Settings → Battery → "Prevent automatic sleeping on power adapter
  when the display is off" + plug in. (The `Amphetamine` app is the GUI version.)
- **Windows:** Settings → System → Power → Screen and sleep → set "When plugged
  in, put my device to sleep" to *Never*. Close the lid action: *Do nothing*.

**2. Suppress Colab's idle prompt (cell below).**
Colab pops up a "Are you still here?" dialog after ~90 min of UI inactivity.
The JS snippet below auto-clicks the connect button every minute. Run it in the
notebook (it injects into the Colab page) **before** starting training.

**3. Trust the resume cell.**
Even with the above, a long run might disconnect (Colab free tier has a hard
12-hour cap; Pro is ~24 h). The `RESUME CELL` in section 5 handles this:
it finds the latest `epoch_NNNN.pt` on Drive and restarts training from there.

**Colab Pro background execution** (paid) lets the notebook keep running with
the tab closed — the cleanest solution if you have access."""


KEEPALIVE_CELL = """# Keep Colab from disconnecting on idle. Paste-and-run; safe to re-execute.
# Auto-clicks the connect button every 60s. Only useful while the tab is open
# in a browser that hasn't been put to sleep by the OS.
from IPython.display import display, Javascript
display(Javascript('''
function ClickConnect() {
  const btn = document.querySelector("colab-connect-button");
  if (btn && btn.shadowRoot) {
    const inner = btn.shadowRoot.querySelector("#connect");
    if (inner) inner.click();
  }
  console.log("colab keep-alive ping " + new Date().toLocaleTimeString());
}
if (window._colabKeepAlive) clearInterval(window._colabKeepAlive);
window._colabKeepAlive = setInterval(ClickConnect, 60000);
console.log("colab keep-alive armed (60s interval)");
'''))
print('Keep-alive armed. Re-run this cell after any browser refresh.')
"""


NEW_INTRO = """# Geometric EEG SSL — Colab Pretraining Notebook

**What this does:**
1. Installs dependencies
2. Mounts Google Drive (checkpoints saved there — survive session disconnects)
3. Arms a keep-alive against laptop sleep + idle disconnect
4. Clones the repo
5. Downloads PhysioNet MI data via MNE — **resume-aware**, safe to re-run after disconnect
6. Pretrains G1 (geometric, 105 subjects, 100 epochs)
7. Runs the linear probe (LOSO, 105 subjects)
8. Optionally: runs G2, G3, transductive codex ablations

**Before running:** Runtime → Change runtime type → GPU (T4 free, A100 Colab Pro)

---"""


def main() -> None:
    with open(NB_PATH) as f:
        nb = json.load(f)

    cells = nb["cells"]
    ids = [c.get("id") for c in cells]

    # Idempotence check
    if "s2b_keepalive_header" in ids and "s2b_keepalive" in ids:
        print("Notebook already updated. No changes made.")
        return

    # 1. Update intro
    intro_idx = next(i for i, c in enumerate(cells) if c.get("id") == "a1b2c3d4")
    cells[intro_idx] = md_cell("a1b2c3d4", NEW_INTRO)

    # 2. Replace s4_download
    s4_idx = next(i for i, c in enumerate(cells) if c.get("id") == "s4_download")
    cells[s4_idx] = code_cell("s4_download", NEW_S4_DOWNLOAD)

    # 3. Insert keep-alive section after s2_drive (Drive mount), before s3_header
    s3_header_idx = next(i for i, c in enumerate(cells) if c.get("id") == "s3_header")
    cells.insert(s3_header_idx, md_cell("s2b_keepalive_header", KEEPALIVE_HEADER))
    cells.insert(s3_header_idx + 1, code_cell("s2b_keepalive", KEEPALIVE_CELL))

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Updated {NB_PATH}")
    print(f"  Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
