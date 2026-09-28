"""The central interior remains measurable, with or without a surviving rim."""
from dataclasses import replace
import numpy as np
import pytest
import tifffile

from tunnel_analysis.config import Config
from tunnel_analysis.io import sha256
from tunnel_analysis.pipeline import run_image
from tunnel_analysis.regions import estimate_geometry, tile_fields, outer_region
from tunnel_analysis.validation import audit_mask, compare_masks


def scene(size=400, rim=True, broken=False):
    yy, xx = np.mgrid[:size, :size]
    cx = cy = size/2
    radius = np.hypot(xx-cx, yy-cy)
    outer = (radius >= .46*size) & (radius <= .465*size)
    inner = (radius >= .115*size) & (radius <= .125*size)
    if broken:
        inner &= ~((xx < cx) & (yy < cy))
    truth = (abs(yy-cy) <= max(1, .005*size)) & (xx > cx-.04*size) & (xx < cx+.04*size)
    # Radial branch joined to the rim; its shared pixels should remain eligible.
    truth |= (abs(yy-cy) <= max(1, .005*size)) & (xx >= cx+.12*size) & (xx < cx+.35*size)
    image = outer | truth | (inner if rim else False)
    if broken:
        truth |= inner  # In this scene the incomplete curved structure is a tunnel.
    return image.astype(np.uint8)*255, truth, inner


@pytest.mark.parametrize('size', [256, 400, 800])
def test_no_rim_does_not_exclude_centre(size):
    image, truth, _ = scene(size, rim=False)
    config = Config()
    geo = estimate_geometry(image.astype(np.float32)/255, image.shape, config)
    region, _ = tile_fields(geo, image.shape, 0, size, 0, size, config)
    yy, xx = np.mgrid[:size, :size]
    outer = outer_region(xx, yy, geo.center_x, geo.center_y, geo.radius_x, geo.radius_y,
                         geo.angle, geo.outer_cutoffs)
    assert geo.details['inner_rim_status'] == 'not_identified'
    assert 'inner_rim_not_identified_review_centre' in geo.flags
    np.testing.assert_array_equal(region, outer)
    assert np.all(region[truth])


@pytest.mark.parametrize('broken', [False, True])
def test_central_tunnels_count_when_rim_missing_or_fragmented(tmp_path, broken):
    image, truth, _ = scene(rim=False, broken=broken)
    # Explicitly include the partial curve when simulating washed-away rim:
    if broken:
        image, truth, _ = scene(broken=True)
    path = tmp_path/'washed.tif'
    tifffile.imwrite(path, image)
    before = sha256(path)
    result = run_image(path, tmp_path/'out', Config(tile_size=96), False)
    assert result['status'] == 'provisional'
    assert result['geometry']['inner_rim_model'] is None
    mask = audit_mask(tmp_path/'out')
    assert result['tunnel_area_px'] == np.count_nonzero(mask)
    metrics = compare_masks(mask, truth)
    assert metrics['precision'] >= .95
    assert metrics['recall'] >= .95
    assert abs(metrics['relative_area_error']) <= .05
    assert sha256(path) == before


def test_surviving_rim_keeps_central_and_adjoining_tunnels(tmp_path):
    image, truth, inner = scene()
    config = Config(tile_size=96)
    geo = estimate_geometry(image.astype(np.float32)/255, image.shape, config)
    region, _ = tile_fields(geo, image.shape, 0, 400, 0, 400, config)
    assert geo.details['inner_rim_status'] == 'detected_requires_review'
    assert region[200, 200]
    assert np.all(region[truth])
    assert np.mean(region[inner & ~truth]) < .05
    path = tmp_path/'rim.tif'; tifffile.imwrite(path, image)
    a = run_image(path, tmp_path/'a', config, False)
    b = run_image(path, tmp_path/'b', replace(config, tile_size=1024), False)
    np.testing.assert_array_equal(audit_mask(tmp_path/'a'), audit_mask(tmp_path/'b'))
    assert a['tunnel_area_px'] == b['tunnel_area_px']
    assert np.all(audit_mask(tmp_path/'a')[truth])


def test_buffer_setting_does_not_enable_exclusion_without_explicit_mode(tmp_path):
    path = tmp_path/'old.json'
    path.write_text('{"inner_buffer_fraction": 0.025}')
    assert Config.read(path).inner_exclusion == "rim-only"


def test_irregular_central_loop_is_not_assumed_to_be_rim(tmp_path):
    size = 400
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx-200, yy-200)
    theta = np.arctan2(yy-200, xx-200)
    curve = abs(radius - (60 + 12*np.sin(3*theta))) <= 2
    outer = (radius >= 184) & (radius <= 186)
    source = tmp_path/'loop.tif'
    tifffile.imwrite(source, ((curve | outer)*255).astype(np.uint8))
    result = run_image(source, tmp_path/'out', Config(), False)
    assert result['geometry']['inner_rim_model'] is None
    metrics = compare_masks(audit_mask(tmp_path/'out'), curve)
    assert metrics['precision'] >= .95
    assert metrics['recall'] >= .95


def test_rim_region_agrees_across_native_tiles():
    image, _, _ = scene()
    config = Config()
    geo = estimate_geometry(image.astype(np.float32)/255, (800, 800), config)
    whole, _ = tile_fields(geo, (800, 800), 0, 800, 0, 800, config)
    tiled = np.zeros_like(whole)
    from tunnel_analysis.segmentation import tiles
    for out, (a, b, c, d), crop in tiles((800, 800), 96):
        region, _ = tile_fields(geo, (800, 800), a, b, c, d, config)
        tiled[out] = region[crop]
    np.testing.assert_array_equal(whole, tiled)
    assert whole[400, 400]
