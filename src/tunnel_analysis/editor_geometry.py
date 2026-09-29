"""Explicit ellipses in native pixel coordinates for interactive review."""
from dataclasses import asdict, dataclass, replace
import math
import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from .regions import Geometry, RegionError, ellipse_radius, estimate_geometry


@dataclass(frozen=True)
class Ellipse:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle: float = 0.0  # radians, image coordinates (y increases downwards)

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Ellipse coordinates must be finite")
        if min(self.radius_x, self.radius_y) <= 0:
            raise ValueError("Ellipse radii must be positive")
        # Matplotlib's selector handles rotation only within +/-45 degrees.
        # Swapping axes yields the same ellipse at an equivalent angle.
        angle = (self.angle + math.pi/2) % math.pi - math.pi/2
        if abs(angle) > math.pi/4:
            a, b = self.radius_x, self.radius_y
            object.__setattr__(self, 'radius_x', b)
            object.__setattr__(self, 'radius_y', a)
            angle += -math.pi/2 if angle > 0 else math.pi/2
        object.__setattr__(self, 'angle', angle)

    def mask(self, x, y):
        return ellipse_mask(self, x, y)


def ellipse_mask(ellipse, x, y):
    return ellipse_radius(x, y, ellipse.center_x, ellipse.center_y,
                          ellipse.radius_x, ellipse.radius_y, ellipse.angle) < 1


@dataclass(frozen=True)
class Boundaries:
    arena: Ellipse
    outer: Ellipse
    inner: Ellipse | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(Ellipse(**data['arena']), Ellipse(**data['outer']),
                   Ellipse(**data['inner']) if data['inner'] else None)

    def masks(self, y0, y1, x0, x1):
        y, x = np.ogrid[y0:y1, x0:x1]
        arena = ellipse_mask(self.arena, x, y)
        permitted = arena & ellipse_mask(self.outer, x, y)
        if self.inner is not None:
            permitted &= ~ellipse_mask(self.inner, x, y)
        return arena, permitted


def suggest_boundaries(image, config):
    """Suggestions only; the operator must confirm or replace them."""
    preview = image.preview(config.preview_size)
    try:
        geometry = estimate_geometry(preview, image.shape,
                                     replace(config, inner_exclusion='rim-only'), image)
        arena = Ellipse(geometry.center_x, geometry.center_y,
                        geometry.radius_x, geometry.radius_y, geometry.angle)
        cutoff = float(np.median(geometry.outer_cutoffs))
        outer = replace(arena, radius_x=arena.radius_x*cutoff, radius_y=arena.radius_y*cutoff)
        return Boundaries(arena, outer), (
            'Automatic outer suggestion; check the arena and outer cutoff. '
            'Inner exclusion is disabled until you draw it.')
    except RegionError as exc:
        h, w = image.shape
        arena = Ellipse((w-1)/2, (h-1)/2, (w-1)/2, (h-1)/2)
        return Boundaries(arena, replace(arena, radius_x=arena.radius_x*.98,
                                         radius_y=arena.radius_y*.98)), (
            f'Automatic fit failed: {exc}. Draw all required boundaries manually.')


def confirmed_geometry(image, boundaries, config):
    """Reuse the detector's background/noise equations with confirmed masks."""
    preview = image.preview(config.preview_size)
    ph, pw = preview.shape
    h, w = image.shape
    y, x = np.ogrid[:ph, :pw]
    x, y = (x+.5)*w/pw-.5, (y+.5)*h/ph-.5
    arena = boundaries.arena
    region = ellipse_mask(arena, x, y) & ellipse_mask(boundaries.outer, x, y)
    inner = np.zeros_like(region) if boundaries.inner is None else ellipse_mask(boundaries.inner, x, y)
    region &= ~inner
    if np.count_nonzero(region) < 16:
        raise ValueError('Confirmed boundaries leave too little image to analyse')
    diameter = 2*np.sqrt(arena.radius_x*arena.radius_y)
    preview_diameter = diameter*np.sqrt(ph*pw/(h*w))
    size = 2*max(1, round(config.background_radius*preview_diameter/1000))+1
    background = ndi.grey_opening(preview, size=(size, size))
    background = ndi.gaussian_filter(background, max(.5, config.background_smooth*preview_diameter/1000))
    vals = np.maximum(preview-background, 0)[region]
    low = vals[vals <= np.median(vals)]
    noise = float(1.4826*np.median(np.abs(low-np.median(low))))
    otsu = float(filters.threshold_otsu(vals)) if np.ptp(vals) > 0 else 1.
    threshold = max(config.intensity_floor, config.otsu_multiplier*otsu, config.noise_multiplier*noise)
    return Geometry(arena.center_x, arena.center_y, arena.radius_x, arena.radius_y,
                    arena.angle, diameter, np.ones(720), inner, background, region,
                    threshold, noise, [], {'notebook_boundaries': boundaries.to_dict(),
                                          'inner_rim_model': None, 'otsu_threshold': otsu})
