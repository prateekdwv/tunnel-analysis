"""Conservative detection of a surviving smooth central rim on the preview.

A missing/irregular boundary is not an opening to reconstruct. The central
interior stays measurable. All detections need biological review: a smooth
closed tunnel loop can be indistinguishable from an experimental rim.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


def detect_inner_rim(preview, arena, rad, cx, cy, diameter, config, fit_ellipse):
    """Return a supported narrow rim model, never a filled central exclusion."""
    h, w = preview.shape
    iy, ix = int(round(cy)), int(round(cx))
    evidence = {'inner_rim_status': 'not_identified', 'inner_rim_model': None,
                'central_interior_included': True,
                'inner_rim_coordinate_system': 'preview pixel centres',
                'inner_rim_preview_shape': [h, w],
                'inner_exclusion_rule': 'supported smooth rim only; no central buffer'}
    # Bright threshold used only for geometry, not tunnel classification.
    bright = preview > config.inner_rim_intensity
    for closing in np.arange(config.central_gap_close, config.central_gap_max + .001,
                             config.central_gap_close):
        barrier = morphology.closing(bright, morphology.disk(max(1, round(closing*diameter/1000))))
        labels, _ = ndi.label((~barrier) & arena)
        window = labels[max(0, iy-3):min(h, iy+4), max(0, ix-3):min(w, ix+4)]
        ids, counts = np.unique(window[window > 0], return_counts=True)
        if not len(ids):
            continue
        proposed = labels == ids[np.argmax(counts)]
        fraction = np.count_nonzero(proposed)/max(1, np.count_nonzero(arena))
        if not (.015 < fraction < .35) or np.any(proposed & (rad > .65)):
            continue
        contours = measure.find_contours(proposed, .5)
        if not contours:
            continue
        xy = max(contours, key=len)[:, ::-1]
        try:
            params, residuals = fit_ellipse(xy)
        except ValueError:
            continue
        ex, ey, a, b, angle = map(float, params)
        if (min(a, b)/max(a, b) < .65 or min(a, b) < .025*diameter or
                max(a, b) > .33*diameter):
            continue
        residual = float(np.quantile(residuals, .9))
        if residual > max(1., config.inner_rim_fit_tolerance*diameter):
            continue
        # Sample short profiles normal-ish to the ellipse. Search only next to
        # the supported boundary, not all the way to distant outward branches.
        theta = np.linspace(0, 2*np.pi, 720, endpoint=False)
        max_width = config.inner_rim_max_width*diameter+2
        offsets = np.arange(-max_width, 2*max_width+1, .5)
        eq_radius = np.sqrt(a*b)
        radii = 1 + offsets[:, None]/eq_radius
        px = ex + radii*(a*np.cos(theta)*np.cos(angle)-b*np.sin(theta)*np.sin(angle))
        py = ey + radii*(a*np.cos(theta)*np.sin(angle)+b*np.sin(theta)*np.cos(angle))
        values = ndi.map_coordinates(preview, [py, px], order=1, mode='constant')
        foreground = values > config.inner_rim_intensity
        starts = np.full(720, np.nan)
        ends = np.full(720, np.nan)
        for i in range(720):
            candidates = np.flatnonzero(foreground[:, i] & (abs(offsets) <= max(1.5, .004*diameter)))
            if not len(candidates):
                continue
            k = candidates[np.argmin(abs(offsets[candidates]))]
            lo = hi = k
            while lo > 0 and foreground[lo-1, i]:
                lo -= 1
            while hi+1 < len(offsets) and foreground[hi+1, i]:
                hi += 1
            if lo == 0 or hi == len(offsets)-1 or offsets[hi]-offsets[lo] > max_width:
                continue
            starts[i], ends[i] = offsets[lo], offsets[hi]
        valid = np.isfinite(starts)
        support = float(np.mean(valid))
        if support < config.inner_rim_min_support:
            continue
        # Stable width/location around the ring; outgoing branches must not
        # enlarge the rejected band. Unsupported angles stay unexcluded.
        typical_start = float(np.median(starts[valid]))
        typical_end = float(np.median(ends[valid]))
        valid &= (abs(starts-typical_start) <= max(1.5, .003*diameter))
        valid &= (abs(ends-typical_end) <= max(1.5, .003*diameter))
        if np.mean(valid) < config.inner_rim_min_support:
            continue
        # Preserve shared ring/branch pixels wherever foreground continues
        # radially beyond the typical rim on either side. This intentionally
        # favours retaining a short uncertain rim fragment at a junction.
        guard = max(1., .001*diameter)
        width = max(1., typical_end-typical_start)
        inside = (offsets < typical_start-guard) & (offsets >= typical_start-guard-2*width)
        outside = (offsets > typical_end+guard) & (offsets <= typical_end+guard+2*width)
        branch = np.zeros(720, bool)
        for band in (inside, outside):
            if np.any(band):
                branch |= np.mean(foreground[band], axis=0) >= .65
        # Angular padding covers antialiasing and junction shoulders.
        padding = max(1, int(np.ceil(720*guard/(2*np.pi*min(a, b)))))
        branch = ndi.maximum_filter1d(branch, 2*padding+1, mode='wrap')
        evidence_support = float(np.mean(valid))
        valid &= ~branch
        rim = {'center_x': ex, 'center_y': ey, 'radius_x': a, 'radius_y': b,
               'angle': angle, 'inner_radius': 1+(typical_start-guard)/eq_radius,
               'outer_radius': 1+(typical_end+guard)/eq_radius,
               'supported_angles': valid.tolist()}
        evidence.update(inner_rim_status='detected_requires_review', inner_rim_model=rim,
                        inner_rim_support_fraction=evidence_support,
                        inner_rim_excluded_angle_fraction=float(np.mean(valid)),
                        inner_rim_fit_residual_preview_px=residual,
                        central_gap_closing_preview_px=max(1, round(closing*diameter/1000)))
        return rim, evidence
    return None, evidence


def rim_mask(x, y, model):
    if model is None:
        return np.zeros(np.broadcast_shapes(x.shape, y.shape), bool)
    dx, dy = x-model['center_x'], y-model['center_y']
    c, s = np.cos(model['angle']), np.sin(model['angle'])
    u = (dx*c+dy*s)/model['radius_x']
    v = (-dx*s+dy*c)/model['radius_y']
    radius = np.hypot(u, v)
    n = len(model['supported_angles'])
    indices = np.rint(np.mod(np.arctan2(v, u), 2*np.pi)*n/(2*np.pi)).astype(int) % n
    return ((radius >= model['inner_radius']) & (radius <= model['outer_radius']) &
            np.asarray(model['supported_angles'], bool)[indices])
