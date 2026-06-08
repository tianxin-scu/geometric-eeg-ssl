# Local GPU driver (Windows / RTX 4070) for the six leave-one-montage-out
# pretrain runs: {geometric, chind} x {no_sleep, no_bcic, no_phys}.
#
# Each run auto-resumes from the latest checkpoint in its ckpt dir, so it is
# safe to Ctrl-C and re-launch. Per-run logs go to runs/pretrain/<name>/train.log.
#
# Usage (all six, serially):
#     .\scripts\run_all.ps1
#
# Run a subset (substring match on the run name):
#     .\scripts\run_all.ps1 -Filter geometric      # all three geometric runs
#     .\scripts\run_all.ps1 -Filter no_bcic        # both variants, BCIC held out
#     .\scripts\run_all.ps1 -Filter chind_no_phys  # one specific run

param(
    [string]$Filter = ""
)

# Keep this at "Continue", NOT "Stop". The training script writes harmless
# warnings (e.g. torch's scheduler.step() deprecation) to stderr. With "Stop"
# + the `2>&1` pipe below, PowerShell promotes any native-stderr line to a
# terminating error and kills the whole queue mid-run. Real failures are
# detected via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Config   = Join-Path $RepoRoot "configs\pretrain\default.yaml"
$CkptRoot = Join-Path $RepoRoot "runs\pretrain"

# Balanced n9 budget (BCIC-2B's 9-subject ceiling sets the shared cap).
# For the n156 headline run, raise -NSubjectsPerDataset to 78 and restrict
# the queue to the no_bcic split (the only one that scales).
$EpochsPerDataset    = 4096
$NSubjectsPerDataset = 9

# (variant, split_tag, pretrain_datasets_csv)
$Runs = @(
    @{ variant = "geometric"; tag = "no_sleep"; datasets = "physionet_mi,bcic_2b"   },
    @{ variant = "geometric"; tag = "no_bcic";  datasets = "physionet_mi,sleep_edfx" },
    @{ variant = "geometric"; tag = "no_phys";  datasets = "bcic_2b,sleep_edfx"     },
    @{ variant = "chind";     tag = "no_sleep"; datasets = "physionet_mi,bcic_2b"   },
    @{ variant = "chind";     tag = "no_bcic";  datasets = "physionet_mi,sleep_edfx" },
    @{ variant = "chind";     tag = "no_phys";  datasets = "bcic_2b,sleep_edfx"     }
)

# Pin cache + MNE paths to the repo root.
$env:EEG_CACHE_DIR = Join-Path $RepoRoot "cache"
$env:MNE_DATA      = Join-Path $RepoRoot "mne_data"

foreach ($r in $Runs) {
    $name = "$($r.variant)_$($r.tag)"
    if ($Filter -ne "" -and $name -notlike "*$Filter*") {
        Write-Output "skip $name (filter=$Filter)"
        continue
    }

    $ckptDir = Join-Path $CkptRoot $name
    New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null
    $logPath = Join-Path $ckptDir "train.log"

    $existing = Get-ChildItem -Path $ckptDir -Filter "epoch_*.pt" -ErrorAction SilentlyContinue
    $mode = if ($existing) { "resuming from $($existing[-1].Name)" } else { "starting fresh" }

    Write-Output ""
    Write-Output "================================================================"
    Write-Output "  $name  ($mode)"
    Write-Output "  pretrain on: $($r.datasets)"
    Write-Output "  ckpt dir:    $ckptDir"
    Write-Output "  log:         $logPath"
    Write-Output "================================================================"

    & $Python -u "$RepoRoot\scripts\pretrain.py" `
        --config $Config `
        --variant $r.variant `
        --pretrain-datasets $r.datasets `
        --epochs-per-dataset $EpochsPerDataset `
        --n-subjects-per-dataset $NSubjectsPerDataset `
        --ckpt-dir $ckptDir `
        --device cuda `
        --resume latest 2>&1 | Tee-Object -FilePath $logPath -Append

    if ($LASTEXITCODE -ne 0) {
        # Don't abort the whole queue on one failure -- each run is independent
        # and resumable via --resume latest. Warn and move to the next.
        Write-Warning "[$name] exited with code $LASTEXITCODE; continuing to next run."
        continue
    }
}

Write-Output ""
Write-Output "all requested runs finished."
