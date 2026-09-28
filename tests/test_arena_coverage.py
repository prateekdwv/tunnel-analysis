from dataclasses import replace
import numpy as np
import tifffile
import pytest

from tunnel_analysis.config import Config
from tunnel_analysis.pipeline import run_image
from tunnel_analysis.regions import estimate_geometry, arena_tile
from tunnel_analysis.validation import synthetic_scene, audit_mask


def test_denominator_does_not_depend_on_rim_rejection(tmp_path):
    image, _, _ = synthetic_scene(320)
    source = tmp_path/'scene.tif'
    tifffile.imwrite(source, image)
    config = Config(preview_size=320)
    baseline = run_image(source, tmp_path/'a', config, False)
    changed = run_image(source, tmp_path/'b', replace(config, outer_inset=20, outer_rim_max=35), False)
    arena = audit_mask(tmp_path/'a', 'analysis_region')
    np.testing.assert_array_equal(arena, audit_mask(tmp_path/'b', 'analysis_region'))
    assert arena[160, 160]
    assert np.any(arena & ~audit_mask(tmp_path/'a', 'segmentation_region'))
    for folder, result in [('a', baseline), ('b', changed)]:
        tunnels = audit_mask(tmp_path/folder)
        eligible = audit_mask(tmp_path/folder, 'segmentation_region')
        assert result['arena_area_px'] == result['analysis_region_area_px'] == np.count_nonzero(arena)
        assert result['tunnel_area_percent'] == pytest.approx(100*np.count_nonzero(tunnels)/np.count_nonzero(arena))
        assert not np.any(tunnels & ~eligible)
        assert not np.any(eligible & ~arena)
        assert result['measurement_rule_version'] == 'arena-coverage-v1'


def test_arena_native_count_agrees_with_analytic_circle_and_tiles():
    image, _, _ = synthetic_scene(256)
    geo = estimate_geometry(image.astype(np.float32)/255, image.shape, Config())
    geo.center_x = geo.center_y = 127.5
    geo.radius_x = geo.radius_y = 115.
    yy, xx = np.mgrid[:256, :256]
    expected = (xx-127.5)**2+(yy-127.5)**2 < 115**2
    whole = arena_tile(geo, 0, 256, 0, 256)
    np.testing.assert_array_equal(whole, expected)
    geo.outer_cutoffs[:] = .8  # changing artifact exclusions cannot change normalization
    geo.details['inner_rim_model'] = None
    tiled = np.zeros_like(whole)
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            tiled[y:y+64, x:x+64] = arena_tile(geo, y, y+64, x, x+64)
    np.testing.assert_array_equal(tiled, whole)


def test_substantially_cropped_arena_is_flagged_without_extrapolation(tmp_path):
    image, _, _ = synthetic_scene(400)
    image = image[:, 55:-55]
    source = tmp_path/'cropped.tif'
    tifffile.imwrite(source, image)
    result = run_image(source, tmp_path/'out', Config(), False)
    assert result['status'] == 'provisional'
    assert 'arena_boundary_clipped_review_coverage' in result['quality_flags']
    assert result['arena_estimated_missing_fraction'] > .005
    arena = audit_mask(tmp_path/'out', 'analysis_region')
    assert result['arena_area_px'] == np.count_nonzero(arena)
    geometry = result['geometry']
    assert result['arena_area_px'] < np.pi*geometry['radius_x']*geometry['radius_y']
