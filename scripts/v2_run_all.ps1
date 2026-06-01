# Local driver for the 6 v2 pretrain runs (mirrors notebook section 7.*).
#
# Each run auto-resumes from the latest checkpoint in its ckpt dir, so it's
# safe to Ctrl-C and re-launch. Saves per-run logs to runs/pretrain/<name>/train.log.
#
# Usage (run all 6 serially):
#     .\scripts\v2_run_all.ps1
#
# Run a subset (substring match on the run name):
#     .\scripts\v2_run_all.ps1 -Filter geometric         # all 3 geometric
#     .\scripts\v2_run_all.ps1 -Filter no_sleep          # both splits with sleep held out
#     .\scripts\v2_run_all.ps1 -Filter v2_chind_no_phys  # one specific run

param(
    [string]$Filter = ""
)

# NOTE: keep this at "Continue", NOT "Stop". The training script writes
# harmless warnings (e.g. torch's scheduler.step() deprecation) to stderr.
# With "Stop" + the `2>&1` pipe below, PowerShell promotes any native-stderr
# line to a terminating error and kills the whole queue mid-run. We detect
# real failures via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Config   = Join-Path $RepoRoot "configs\v2\pretrain\v2_default.yaml"
$CkptRoot = Join-Path $RepoRoot "runs\pretrain"

# Match colab notebook: section 6 budget.
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

# Pretrained env: cache + mne paths.
$env:EEG_CACHE_DIR = Join-Path $RepoRoot "cache"
$env:MNE_DATA      = Join-Path $RepoRoot "mne_data"

foreach ($r in $Runs) {
    $name = "v2_$($r.variant)_$($r.tag)"
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

    & $Python -u "$RepoRoot\scripts\v2_pretrain.py" `
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
