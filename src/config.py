"""Project-wide configuration.

Three tiers (see project_direction_summary.md and the design discussion):

- Tier 1: locked preprocessing choices. Changing these invalidates cross-dataset
  comparability; bump the config version rather than editing in place.
- Tier 2: ablation knobs that map to proposal section 4 (O1/O2/O3/O4).
- Tier 3: artifact policy, deliberately asymmetric between pretrain and eval.

Every run should log the *resolved* config (see `resolved_dict`) alongside its
checkpoint so future-me can reconstruct what produced any given number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple

import yaml


# ---------------------------------------------------------------------------
# Tier 1 -- locked preprocessing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """Locked preprocessing. Identical across all four candidate datasets."""

    bandpass_hz: Tuple[float, float] = (0.5, 45.0)
    resample_hz: int = 200
    epoch_length_s: float = 4.0
    detrend: str = "linear"              # "linear" | "constant" | "none"
    normalize: str = "per_epoch_zscore"  # only option implemented
    reference: str = "native"            # keep dataset-shipped reference
    ica: bool = False
    montage: str = "standard_1005"
    coord_normalize: bool = True         # centroid-subtract + unit-scale

    def __post_init__(self) -> None:
        if self.bandpass_hz[0] >= self.bandpass_hz[1]:
            raise ValueError(f"bandpass low >= high: {self.bandpass_hz}")
        if self.resample_hz <= 2 * self.bandpass_hz[1]:
            raise ValueError(
                f"resample_hz={self.resample_hz} violates Nyquist for "
                f"bandpass high={self.bandpass_hz[1]}"
            )
        if self.epoch_length_s <= 0:
            raise ValueError(f"epoch_length_s must be positive: {self.epoch_length_s}")
        if self.detrend not in {"linear", "constant", "none"}:
            raise ValueError(f"unknown detrend: {self.detrend}")
        if self.normalize != "per_epoch_zscore":
            raise NotImplementedError(f"normalize={self.normalize} not implemented")

    @property
    def samples_per_epoch(self) -> int:
        return int(round(self.epoch_length_s * self.resample_hz))


# ---------------------------------------------------------------------------
# Tier 2 -- ablation knobs (proposal section 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationConfig:
    """Knobs the proposal explicitly varies. Defaults match section 4."""

    # O1: where g_ij enters attention. "G1" = score-only (default).
    # "none" = vanilla attention (channel-independent baseline, no geometry).
    geometry_injection: str = "G1"  # "G1" | "G2" | "G3" | "none"

    # O2: codex on/off. False = geometric encoder (default, headline).
    use_codex: bool = False

    # O3: reconstruction loss weight. 1.0 = on (default).
    lambda_recon: float = 1.0

    # O4: masking.
    mask_ratio: float = 0.5
    mask_scheme: str = "joint"  # "joint" | "temporal" | "spatial"

    def __post_init__(self) -> None:
        if self.geometry_injection not in {"G1", "G2", "G3", "none"}:
            raise ValueError(f"geometry_injection must be G1/G2/G3/none: {self.geometry_injection}")
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio out of range: {self.mask_ratio}")
        if self.mask_scheme not in {"joint", "temporal", "spatial"}:
            raise ValueError(f"unknown mask_scheme: {self.mask_scheme}")
        if self.lambda_recon < 0:
            raise ValueError(f"lambda_recon must be >= 0: {self.lambda_recon}")


# ---------------------------------------------------------------------------
# Architectural defaults (proposal section 3)
# ---------------------------------------------------------------------------
#
# These are NOT ablation knobs -- they're the architecture's fixed shape per
# proposal section 3 (L_S=2, L_T=4, H=8, d=256). They live here rather than
# being hard-coded in model classes so a single resolved config (logged next
# to a checkpoint) captures the exact architecture that produced any number.
# Changing these between geometric and transductive runs would invalidate the
# headline comparison; the headline runs share a single ArchConfig.

@dataclass(frozen=True)
class ArchConfig:
    """Architectural defaults from proposal section 3.

    Shared between the geometric encoder and the transductive baseline so
    the headline comparison is parameter-matched on everything except the
    distinguishing component (codex vs. geometric MLPs).
    """

    # Width and head count (shared between spatial and temporal transformers
    # by convention; split into separate fields in case I want to break the
    # tie later).
    d_model: int = 256
    n_heads_spatial: int = 8
    n_layers_spatial: int = 2
    n_heads_temporal: int = 8
    n_layers_temporal: int = 4
    mlp_ratio: float = 4.0

    # Geometric attention's inner MLP hidden width (R^4 -> hidden -> H or d).
    geom_mlp_hidden: int = 32

    # Patching. Defaults: 250 ms at 200 Hz = 50 samples, non-overlapping.
    # patch_stride == patch_samples means non-overlapping (session 4 decision).
    patch_samples: int = 50
    patch_stride: int = 50

    # Temporal positional embedding. "learned" per proposal section 3.
    # "sinusoidal" reserved for a future ablation; "none" disables.
    temporal_posemb: str = "learned"

    # Dropout. Single knob shared across attention and MLP for simplicity;
    # split if I want finer control later.
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive: {self.d_model}")
        if self.d_model % self.n_heads_spatial != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by "
                f"n_heads_spatial={self.n_heads_spatial}"
            )
        if self.d_model % self.n_heads_temporal != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by "
                f"n_heads_temporal={self.n_heads_temporal}"
            )
        if self.n_layers_spatial <= 0 or self.n_layers_temporal <= 0:
            raise ValueError(
                f"layer counts must be positive: "
                f"L_S={self.n_layers_spatial}, L_T={self.n_layers_temporal}"
            )
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive: {self.mlp_ratio}")
        if self.geom_mlp_hidden <= 0:
            raise ValueError(f"geom_mlp_hidden must be positive: {self.geom_mlp_hidden}")
        if self.patch_samples <= 0:
            raise ValueError(f"patch_samples must be positive: {self.patch_samples}")
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive: {self.patch_stride}")
        if self.patch_stride > self.patch_samples:
            raise ValueError(
                f"patch_stride={self.patch_stride} > "
                f"patch_samples={self.patch_samples} would skip samples"
            )
        if self.temporal_posemb not in {"learned", "sinusoidal", "none"}:
            raise ValueError(f"unknown temporal_posemb: {self.temporal_posemb}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout out of range: {self.dropout}")

    def n_patches(self, samples_per_epoch: int) -> int:
        """Number of patches per epoch given a sample count.

        Cross-checked against PreprocessConfig.samples_per_epoch at the
        top-level Config validator so the architecture and preprocessing
        agree on T_p.
        """
        if samples_per_epoch < self.patch_samples:
            raise ValueError(
                f"samples_per_epoch={samples_per_epoch} < "
                f"patch_samples={self.patch_samples}"
            )
        # Floor division: any trailing samples shorter than a full patch
        # are dropped. With the default non-overlapping stride==samples
        # and the default 4 s / 200 Hz / 50 samples this gives exactly 16.
        return 1 + (samples_per_epoch - self.patch_samples) // self.patch_stride


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters for the pretraining loop.

    tau_base / tau_final define the cosine EMA schedule (EEGPT-style):
        tau(k) = 1 - (1 - tau_base) * (cos(pi * k / K) + 1) / 2
    which ramps from tau_base at step 0 to tau_final at step K.
    Use PretrainModel.cosine_tau(step, total_steps, tau_base, tau_final)
    to compute tau at each step.
    """

    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    n_epochs: int = 100
    lr_warmup_epochs: int = 10   # linear LR warmup before cosine decay

    # EMA schedule: cosine ramp from tau_base to tau_final over training.
    # EEGPT uses 0.996 → 1.0; the fixed-0.99 value in the BYOL paper is
    # a reasonable alternative but gives a flatter momentum trajectory.
    tau_base: float = 0.996
    tau_final: float = 1.0

    # Reconstruction loss weight (mirrors AblationConfig.lambda_recon for
    # convenience so the training loop reads one config section).
    lambda_R: float = 1.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive: {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0: {self.weight_decay}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive: {self.n_epochs}")
        if self.lr_warmup_epochs < 0:
            raise ValueError(f"lr_warmup_epochs must be >= 0: {self.lr_warmup_epochs}")
        if not (0.0 < self.tau_base < 1.0):
            raise ValueError(f"tau_base must be in (0, 1): {self.tau_base}")
        if not (self.tau_base <= self.tau_final <= 1.0):
            raise ValueError(
                f"tau_final must be in [tau_base, 1.0]: "
                f"tau_base={self.tau_base}, tau_final={self.tau_final}"
            )
        if self.lambda_R < 0:
            raise ValueError(f"lambda_R must be >= 0: {self.lambda_R}")


# ---------------------------------------------------------------------------
# Tier 3 -- artifact policy (asymmetric pretrain vs eval)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PretrainArtifactConfig:
    """Loose policy for SSL pretraining: clip extreme values, keep everything."""
    clip_sigma: float = 20.0


@dataclass(frozen=True)
class EvalArtifactConfig:
    """Artifact policy for downstream linear-probe evaluation.

    amplitude_reject_uv: peak-amplitude threshold in µV. 0.0 = disabled.
    Disabled by default: PhysioNet MI raw amplitudes routinely exceed 100 µV
    (measured: up to ~600 µV after bandpass + resample), so any finite threshold
    discards the majority of epochs. EEGPT makes the same choice — no amplitude
    rejection, rely on per-epoch z-score normalization to suppress outliers.
    """
    amplitude_reject_uv: float = 0.0   # 0 = disabled (EEGPT stance)
    autoreject: bool = False  # opt-in; turn on if per-subject variance is high


@dataclass(frozen=True)
class ArtifactConfig:
    pretrain: PretrainArtifactConfig = field(default_factory=PretrainArtifactConfig)
    eval: EvalArtifactConfig = field(default_factory=EvalArtifactConfig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    version: str = "v1"

    def __post_init__(self) -> None:
        # Cross-tier sanity: architecture's patching must agree with
        # preprocessing's epoch length. Trip this early rather than failing
        # at the first forward pass with a confusing shape mismatch.
        T = self.preprocess.samples_per_epoch
        if T < self.arch.patch_samples:
            raise ValueError(
                f"samples_per_epoch={T} < patch_samples="
                f"{self.arch.patch_samples}; no patches would be produced"
            )
        n_patches = self.arch.n_patches(T)
        consumed = self.arch.patch_samples + (n_patches - 1) * self.arch.patch_stride
        if consumed != T:
            raise ValueError(
                f"patching leaves {T - consumed} unused samples "
                f"(T={T}, patch_samples={self.arch.patch_samples}, "
                f"patch_stride={self.arch.patch_stride}, n_patches={n_patches}); "
                f"either adjust patch_stride/patch_samples or accept a "
                f"partial trailing window by relaxing this check."
            )

    def resolved_dict(self) -> dict:
        """Full config as a dict, for logging next to checkpoints."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.resolved_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        pp = dict(data["preprocess"])
        # YAML round-trips tuples as lists; coerce back.
        if "bandpass_hz" in pp:
            pp["bandpass_hz"] = tuple(pp["bandpass_hz"])
        # arch section is new in session 5; fall back to defaults for older
        # YAMLs written before it existed.
        arch_data = data.get("arch", {})
        return cls(
            preprocess=PreprocessConfig(**pp),
            arch=ArchConfig(**arch_data),
            ablation=AblationConfig(**data["ablation"]),
            artifact=ArtifactConfig(
                pretrain=PretrainArtifactConfig(**data["artifact"]["pretrain"]),
                eval=EvalArtifactConfig(**data["artifact"]["eval"]),
            ),
            train=TrainConfig(**data.get("train", {})),
            version=data.get("version", "v1"),
        )