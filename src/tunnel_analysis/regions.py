"""Automatic arena geometry and background estimation on small images."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage import measure, morphology, filters
from .inner_rim import detect_inner_rim, rim_mask


class RegionError(ValueError):
    pass


@dataclass
class Geometry:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle: float
    diameter: float
    outer_cutoffs: np.ndarray
    inner_mask: np.ndarray
    background: np.ndarray
    preview_region: np.ndarray
    threshold: float
    noise: float
    flags: list
    details: dict

    def describe(self):
        return {k: getattr(self, k) for k in ("center_x", "center_y", "radius_x", "radius_y", "angle", "diameter", "threshold", "noise")} | self.details


def ellipse_radius(x, y, cx, cy, a, b, angle):
    dx, dy = x - cx, y - cy
    c, s = np.cos(angle), np.sin(angle)
    return np.sqrt(((dx*c + dy*s)/a)**2 + ((-dx*s + dy*c)/b)**2)


def fit_ellipse(xy):
    """Bounded geometric fit, avoiding degenerate algebraic eigen-solutions."""
    origin = np.mean(xy,axis=0)
    unit = float(np.max(np.ptp(xy,axis=0)))
    if unit <= 0:
        raise RegionError("Degenerate boundary")
    normalized = (xy-origin)/unit
    radius = float(np.median(np.linalg.norm(normalized,axis=1)))
    fit = least_squares(lambda p: ellipse_radius(normalized[:,0],normalized[:,1],*p)-1,
        [0,0,radius,radius,0],bounds=([-1,-1,.02,.02,-np.pi],[1,1,2,2,np.pi]),
        max_nfev=150,loss="soft_l1",f_scale=.02)
    if not fit.success:
        raise RegionError("Ellipse refinement failed")
    cx,cy,a,b,t = fit.x
    params = (cx*unit+origin[0],cy*unit+origin[1],a*unit,b*unit,t)
    residual = np.abs(ellipse_radius(xy[:,0],xy[:,1],*params)-1)*np.sqrt(a*b)*unit
    return params,residual


def outer_region(x,y,cx,cy,a,b,angle,cutoffs):
    c,s = np.cos(angle),np.sin(angle)
    u = ((x-cx)*c+(y-cy)*s)/a
    v = (-(x-cx)*s+(y-cy)*c)/b
    theta = np.mod(np.arctan2(v,u),2*np.pi)
    cutoff = np.interp(theta,np.linspace(0,2*np.pi,len(cutoffs),endpoint=False),cutoffs,period=2*np.pi)
    return u*u+v*v < cutoff*cutoff


def estimate_geometry(preview, native_shape, config, image=None):
    h, w = preview.shape
    yy, xx = np.ogrid[:h, :w]
    support = preview > max(0.01, float(preview.max()) * 0.025)
    if np.count_nonzero(support) < 100:
        raise RegionError("Too little foreground to identify an arena")
    hull = morphology.convex_hull_image(support)
    contours = measure.find_contours(np.pad(hull, 1), 0.5)
    contour = max(contours, key=len) - 1
    xy = contour[:, ::-1][::max(1, len(contour)//2000)]
    # Circle consensus avoids unstable five-point ellipse fits on nearly
    # collinear boundary samples; refine the consensus with a full ellipse.
    consensus, inliers = measure.ransac(xy, measure.CircleModel, min_samples=3,
        residual_threshold=max(2.0, max(h,w)*0.012), max_trials=150, rng=0)
    if consensus is None:
        raise RegionError("Could not fit the outer arena boundary")
    params,_ = fit_ellipse(xy[inliers])
    cx, cy, a, b, angle = map(float, params)
    diameter = 2 * np.sqrt(a*b)
    if (min(a,b)/max(a,b) < 0.75 or diameter < min(h,w)*0.55 or
            diameter > max(h,w)*1.12 or not (0 <= cx < w and 0 <= cy < h)):
        raise RegionError("Outer arena fit is implausible")
    fit_fraction = float(np.mean(inliers))
    flags = []
    if fit_fraction < 0.8:
        flags.append("outer_boundary_uncertain")
    sy, sx = native_shape[0]/h, native_shape[1]/w
    # Refine the fitted scale using native-resolution samples across the boundary.
    # Only rays with an actual outside-to-inside contrast contribute.
    refinement = 0.0
    if image is not None:
        shifts = np.arange(-3, 3.01, 0.25) * diameter/1000
        theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
        c, s = np.cos(angle), np.sin(angle)
        offsets = []
        for t in theta:
            ex, ey = a*np.cos(t)*c-b*np.sin(t)*s, a*np.cos(t)*s+b*np.sin(t)*c
            radii = 1 + shifts/(diameter/2)
            px = np.rint((cx + radii*ex)*sx).astype(int)
            py = np.rint((cy + radii*ey)*sy).astype(int)
            if np.any((px < 0)|(px >= native_shape[1])|(py < 0)|(py >= native_shape[0])):
                continue
            values = image.array[py,px].astype(np.float32)
            if values.ndim == 2:
                values = values @ np.array([.2126,.7152,.0722], np.float32)
            values /= image.maximum
            if image.invert:
                values = 1-values
            drops = -np.diff(ndi.gaussian_filter1d(values, 1))
            j = int(np.argmax(drops))
            if drops[j] > .02:
                offsets.append(float((shifts[j]+shifts[j+1])/2))
        if len(offsets) >= 30:
            refinement = float(np.median(offsets))
            a *= 1+2*refinement/diameter
            b *= 1+2*refinement/diameter
            diameter = 2*np.sqrt(a*b)
    rad = ellipse_radius(xx, yy, cx, cy, a, b, angle)
    # Follow smooth bright outer rims in polar coordinates. Median smoothing
    # suppresses narrow radial tunnels so they do not dictate the rim band.
    angles = np.linspace(0,2*np.pi,720,endpoint=False)
    ds = np.linspace(0,config.outer_rim_max*diameter/1000,96)
    depths = np.full(720,config.outer_inset*diameter/1000)
    for i,t in enumerate(angles):
        radii = 1-2*ds/diameter
        px = cx+radii*(a*np.cos(t)*np.cos(angle)-b*np.sin(t)*np.sin(angle))
        py = cy+radii*(a*np.cos(t)*np.sin(angle)+b*np.sin(t)*np.cos(angle))
        vals = ndi.map_coordinates(preview,[py,px],order=1,mode="constant")
        bright = np.flatnonzero(vals > .2)
        if len(bright) and ds[bright[0]] <= min(config.outer_rim_max,2*config.outer_inset)*diameter/1000:
            end = bright[0]
            while end+1 < len(vals) and vals[end+1] > .12:
                end += 1
            depths[i] = max(depths[i],ds[end]+diameter*.001)
    depths = ndi.median_filter(depths,size=15,mode="wrap")
    depths = ndi.gaussian_filter1d(depths,2,mode="wrap")
    cutoffs = 1-2*depths/diameter
    if np.mean(depths >= config.outer_rim_max*diameter/1000) > .1:
        flags.append("outer_rim_width_at_limit")
    arena = outer_region(xx,yy,cx,cy,a,b,angle,cutoffs)
    if config.inner_exclusion == "opening-buffer":
        iy, ix = int(round(cy)), int(round(cx))
        inner = None
        for closing in np.arange(config.central_gap_close,config.central_gap_max+.001,config.central_gap_close):
            close_radius = max(1,round(closing*diameter/1000))
            barrier = morphology.closing(preview > .025,morphology.disk(close_radius))
            labels,_ = ndi.label((~barrier)&arena)
            seed_window = labels[max(0,iy-3):iy+4,max(0,ix-3):ix+4]
            ids,counts = np.unique(seed_window[seed_window > 0],return_counts=True)
            if not len(ids):
                continue
            proposed = labels == ids[np.argmax(counts)]
            fraction = np.count_nonzero(proposed)/np.count_nonzero(arena)
            if .015 < fraction < .35 and not np.any(proposed & (rad > .65)):
                inner = proposed
                break
        if inner is None:
            raise RegionError("Central opening is unbounded or implausible")
        if closing > config.central_gap_close:
            flags.append("central_boundary_required_gap_closing")
        inner = ndi.binary_fill_holes(inner)
        # The same geometric buffer applies to irregular and circular openings.
        # Retain the bare opening separately for native-resolution refinement.
        buffer_preview = config.inner_buffer_fraction*diameter
        buffered_inner = ndi.distance_transform_edt(~inner) <= buffer_preview
        region = arena & ~buffered_inner
        inner_details = {"inner_rim_model": None, "central_opening_fraction": float(fraction),
                         "central_interior_included": False,
                         "central_gap_closing_px": float(close_radius*np.sqrt(sx*sy)),
                         "inner_buffer_fraction": config.inner_buffer_fraction,
                         "inner_buffer_px": float(config.inner_buffer_fraction*diameter*np.sqrt(sx*sy)),
                         "inner_exclusion_rule": "opening plus fixed Euclidean buffer"}
    else:
        rim, inner_details = detect_inner_rim(preview, arena, rad, cx, cy, diameter, config, fit_ellipse)
        inner = rim_mask(xx, yy, rim)
        region = arena & ~inner
        flags.append("inner_rim_exclusion_requires_review" if rim else "inner_rim_not_identified_review_centre")
    size = 2*max(1,round(config.background_radius*diameter/1000))+1
    # Separable square opening is O(N), unlike a large rank filter with a disk.
    background = ndi.grey_opening(preview, size=(size,size))
    background = ndi.gaussian_filter(background, max(.5,config.background_smooth*diameter/1000))
    corrected = np.maximum(preview-background,0)
    vals = corrected[region]
    # Estimate noise from the lower half, avoiding bright foreground.
    low = vals[vals <= np.median(vals)]
    noise = float(1.4826*np.median(np.abs(low-np.median(low))))
    otsu = float(filters.threshold_otsu(vals)) if np.ptp(vals) > 0 else 1.0
    threshold = max(config.intensity_floor, config.otsu_multiplier*otsu, config.noise_multiplier*noise)
    return Geometry(cx*sx, cy*sy, a*sx, b*sy, angle, diameter*np.sqrt(sx*sy),
                    cutoffs, inner, background, region, threshold, noise, flags,
                    {"outer_fit_inlier_fraction": fit_fraction, "native_boundary_refinement_px": refinement*np.sqrt(sx*sy),
                     **inner_details,
                     "outer_rim_depth_range_px": [float(depths.min()*np.sqrt(sx*sy)),float(depths.max()*np.sqrt(sx*sy))],
                     "outer_inset_px": float(config.outer_inset*diameter*np.sqrt(sx*sy)),
                     "otsu_threshold": otsu})


def tile_fields(geometry, native_shape, y0, y1, x0, x1, config, native_gray=None, background_only=False):
    yy, xx = np.mgrid[y0:y1,x0:x1].astype(np.float32)
    ph,pw = geometry.inner_mask.shape
    coords = [(yy+.5)*ph/native_shape[0]-.5, (xx+.5)*pw/native_shape[1]-.5]
    background = ndi.map_coordinates(geometry.background,coords,order=1,mode="nearest")
    if background_only:
        return None,background
    if 'notebook_boundaries' in geometry.details:
        from .editor_geometry import Boundaries
        _, region = Boundaries.from_dict(geometry.details['notebook_boundaries']).masks(y0, y1, x0, x1)
        return region, background
    if config.inner_exclusion == "opening-buffer":
        inner = ndi.map_coordinates(geometry.inner_mask.astype(np.uint8),coords,order=0,mode="nearest") > 0
        if native_gray is not None:
            # Refine the one-preview-pixel uncertainty band with native intensities.
            # The opening interior remains excluded; dilation follows refinement.
            width = max(1,int(np.ceil(max(native_shape[0]/ph,native_shape[1]/pw))))
            core = ndi.minimum_filter(inner,size=2*width+1)
            band = ndi.maximum_filter(inner,size=2*width+1)
            inner = core | (band & (native_gray < .025))
        if np.any(inner):
            inner = ndi.distance_transform_edt(~inner) <= config.inner_buffer_fraction*geometry.diameter
    else:
        inner = rim_mask(coords[1], coords[0], geometry.details["inner_rim_model"])
    region = outer_region(xx,yy,geometry.center_x,geometry.center_y,geometry.radius_x,geometry.radius_y,geometry.angle,geometry.outer_cutoffs) & ~inner
    return region, background


def arena_tile(geometry, y0, y1, x0, x1):
    """Denominator mask: observed native pixel centres inside the fitted arena.

    Independent of inner/outer rim cutoffs, candidate masks, and classification.
    Cropping is flagged separately; no unseen pixels are extrapolated.
    """
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return ellipse_radius(xx, yy, geometry.center_x, geometry.center_y,
                          geometry.radius_x, geometry.radius_y, geometry.angle) < 1
