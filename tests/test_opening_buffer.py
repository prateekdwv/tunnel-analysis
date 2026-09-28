from dataclasses import replace
import numpy as np
import pytest
from scipy import ndimage as ndi
import tifffile
from skimage import morphology
from tunnel_analysis.config import Config
from tunnel_analysis.regions import estimate_geometry, ellipse_radius, outer_region, RegionError
from tunnel_analysis.pipeline import run_image
from tunnel_analysis.validation import audit_mask
from test_inner_rim import scene


@pytest.mark.parametrize('irregular', [False, True])
def test_restored_opening_buffer_matches_legacy(irregular):
    size=400
    yy,xx=np.mgrid[:size,:size]
    radius=np.hypot(xx-200,yy-200)
    angle=np.arctan2(yy-200,xx-200)
    curve=abs(radius-(60+12*np.sin(3*angle) if irregular else 60))<=2
    image=((curve | ((radius>=184)&(radius<=186)))*1.).astype(np.float32)
    cfg=Config(inner_exclusion='opening-buffer')
    g=estimate_geometry(image,image.shape,cfg)
    arena=outer_region(xx,yy,g.center_x,g.center_y,g.radius_x,g.radius_y,g.angle,g.outer_cutoffs)
    rad=ellipse_radius(xx,yy,g.center_x,g.center_y,g.radius_x,g.radius_y,g.angle)
    expected=None
    for close in np.arange(cfg.central_gap_close,cfg.central_gap_max+.001,cfg.central_gap_close):
        labels,_=ndi.label(~morphology.closing(image>.025,morphology.disk(max(1,round(close*g.diameter/1000)))) & arena)
        ids,counts=np.unique(labels[197:204,197:204],return_counts=True)
        valid=ids>0
        proposed=labels==ids[valid][np.argmax(counts[valid])]
        fraction=proposed.sum()/arena.sum()
        if .015<fraction<.35 and not np.any(proposed & (rad>.65)):
            expected=ndi.binary_fill_holes(proposed)
            break
    np.testing.assert_array_equal(g.inner_mask,expected)
    np.testing.assert_array_equal(g.preview_region,arena & (ndi.distance_transform_edt(~expected)>.025*g.diameter))
    assert g.details['inner_buffer_fraction']==.025


def test_buffer_native_masks_tiling_denominator_and_failure(tmp_path):
    image,_,_=scene()
    path=tmp_path/'ring.tif';tifffile.imwrite(path,image)
    cfg=Config(inner_exclusion='opening-buffer',tile_size=96)
    a=run_image(path,tmp_path/'a',cfg,False)
    b=run_image(path,tmp_path/'b',replace(cfg,tile_size=1024),False)
    original=run_image(path,tmp_path/'original',Config(),False)
    assert a['status']=='provisional'
    np.testing.assert_array_equal(audit_mask(tmp_path/'a'),audit_mask(tmp_path/'b'))
    assert a['arena_area_px']==original['arena_area_px']
    import io,zipfile
    with zipfile.ZipFile(tmp_path/'a'/'audit.zip') as archive:
        region=tifffile.imread(io.BytesIO(archive.read(a['masks']['segmentation_region']['archive_path'])))>0
    tunnels=audit_mask(tmp_path/'a')
    assert not np.any(tunnels & ~region)
    assert not region[200,200]
    assert a['tunnel_area_px']==np.count_nonzero(tunnels)
    missing,_,_=scene(rim=False)
    with pytest.raises(RegionError,match='Central opening'):
        estimate_geometry(missing.astype(np.float32)/255,missing.shape,cfg)


def test_ring_only_has_zero_measured_area(tmp_path):
    size=256
    yy,xx=np.mgrid[:size,:size]
    radius=np.hypot(xx-size/2,yy-size/2)
    image=(((abs(radius-35)<=2) | ((radius>=118)&(radius<=120)))*255).astype(np.uint8)
    path=tmp_path/'ring only.tif';tifffile.imwrite(path,image)
    result=run_image(path,tmp_path/'out',Config(inner_exclusion='opening-buffer'),False)
    assert result['status']=='provisional'
    assert result['tunnel_area_px']==0
