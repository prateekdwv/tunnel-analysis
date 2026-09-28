"""Algorithm settings. Most lengths use pixels per 1,000 arena diameter pixels.

inner_rim_fit_tolerance and inner_rim_max_width are fractions of arena diameter.
"""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class Config:
    preview_size: int = 1400
    tile_size: int = 1024
    outer_inset: float = 8.0
    outer_rim_max: float = 24.0
    central_gap_close: float = 2.0
    central_gap_max: float = 8.0
    inner_rim_intensity: float = 0.12
    inner_rim_fit_tolerance: float = 0.003
    inner_rim_max_width: float = 0.012
    inner_rim_min_support: float = 0.85
    background_radius: float = 12.0
    background_smooth: float = 4.0
    ridge_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    ridge_beta: float = 0.5
    ridge_gamma: float = 0.06
    ridge_threshold: float = 0.12
    intensity_floor: float = 0.06
    otsu_multiplier: float = 0.65
    noise_multiplier: float = 4.0
    recovery_radius: float = 3.0
    max_half_width: float = 8.0
    blob_intensity: float = 0.65
    min_component_area: float = 6.0
    min_elongation: float = 2.5
    min_network_extent: float = 35.0
    rejection_warning: float = 0.6
    sensitivity_warning: float = 0.15

    def __post_init__(self):
        for name in ("preview_size", "tile_size"):
            v = getattr(self, name)
            if type(v) is not int or v < 64:
                raise ValueError(f"{name} must be an integer >= 64")
        for f in fields(self):
            if f.name in ("preview_size", "tile_size", "ridge_sigmas"):
                continue
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{f.name} must be finite and positive")
        if not self.ridge_sigmas or any(not math.isfinite(s) or s <= 0 for s in self.ridge_sigmas):
            raise ValueError("ridge_sigmas must contain positive finite numbers")
        for name in ("ridge_threshold", "intensity_floor", "blob_intensity", "inner_rim_intensity", "inner_rim_fit_tolerance", "inner_rim_max_width", "inner_rim_min_support", "rejection_warning", "sensitivity_warning"):
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be less than 1")
        if self.central_gap_max < self.central_gap_close or self.outer_rim_max < self.outer_inset:
            raise ValueError("Maximum geometry widths must be at least their minimum widths")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path: str | Path | None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        if "ridge_sigmas" in data:
            data["ridge_sigmas"] = tuple(data["ridge_sigmas"])
        return cls(**data)
