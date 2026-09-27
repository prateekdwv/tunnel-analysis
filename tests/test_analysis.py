from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import tifffile
from tunnel_analysis.config import Config
from tunnel_analysis.io import InputImage
from tunnel_analysis.regions import estimate_geometry,RegionError,tile_fields,outer_region
from tunnel_analysis.segmentation import ridge_response,tiles,segment,counts,reconcile_components,suppress_wide_patches
from tunnel_analysis.validation import synthetic_scene,synthetic_reference,audit_mask,compare_masks,validate_crops
from tunnel_analysis.pipeline import run_image
from tunnel_analysis.cli import main


def test_counting_and_reference_metrics():
    mask=np.array([[1,0,0],[0,1,0]],np.uint8)
    region=np.ones_like(mask)
    candidate=np.array([[1,1,0],[0,1,0]],np.uint8)
    result=counts(mask,region,candidate)
    assert result["tunnel_area_px"] == 2
    assert result["analysis_region_area_px"] == 6
    assert result["tunnel_area_percent"] == pytest.approx(100/3)
    comparison=compare_masks(mask,candidate)
    assert comparison["precision"] == 1
    assert comparison["recall"] == pytest.approx(2/3)
    assert comparison["dice"] == .8
    assert compare_masks(mask*0,mask*0)["iou"] == 1
    with pytest.raises(ValueError):
        compare_masks(mask,mask[:1])


def test_rgb_uint16_and_stack_rejection(tmp_path):
    arr=np.zeros((80,90,3),np.uint16)
    arr[...,0]=65535
    tifffile.imwrite(tmp_path/"rgb.tif",arr,photometric="rgb")
    im=InputImage(tmp_path/"rgb.tif")
    assert im.gray().dtype == np.float32
    np.testing.assert_allclose(im.gray(),.2126,atol=1e-6)
    im.close()
    tifffile.imwrite(tmp_path/"stack.tif",np.zeros((4,80,90),np.uint8),photometric="minisblack")
    with pytest.raises(ValueError,match="stacks"):
        InputImage(tmp_path/"stack.tif")


def test_config_rejects_invalid_values(tmp_path):
    with pytest.raises(ValueError): Config(tile_size=0)
    with pytest.raises(ValueError): Config(ridge_gamma=float("nan"))
    with pytest.raises(ValueError): Config(ridge_sigmas=())
    p=tmp_path/"bad.json";p.write_text('{"misspelt_threshold": 1}')
    with pytest.raises(ValueError,match="Unknown"): Config.read(p)


def test_finite_support_ridges_match_across_tiles():
    image,_,_=synthetic_scene(240)
    image=image.astype(np.float32)/255
    sigmas=[.7,1.2,2.5]
    whole=ridge_response(image,sigmas,.5,.06)
    tiled=np.empty_like(whole)
    for out,(a,b,c,d),crop in tiles(image.shape,64,12):
        tiled[out]=ridge_response(image[a:b,c:d],sigmas,.5,.06)[crop]
    np.testing.assert_allclose(tiled,whole,atol=1e-7,rtol=1e-6)


def test_global_components_retain_disconnected_curves_and_reject_dots(tmp_path):
    mask=np.zeros((256,256),np.uint8)
    mask[100:103,10:245]=1  # crosses several tile seams
    mask[30:33,30:80]=1    # disconnected fragment
    yy,xx=np.mgrid[:256,:256]
    dot=(xx-120)**2+(yy-170)**2 < 6**2
    mask[dot]=1
    reconcile_components(mask,1000,Config(),tmp_path)
    assert np.all(mask[100:103,10:245])
    assert np.all(mask[30:33,30:80])
    assert not np.any(mask[dot])


def test_wide_blob_suppression_preserves_branch_junction():
    yy,xx=np.mgrid[:160,:160]
    blob=(xx-80)**2+(yy-80)**2 <= 14**2
    attached=blob.copy();attached[78:83,10:81]=True
    suppressed,_=suppress_wide_patches(attached,5)
    assert np.count_nonzero(suppressed & blob) > np.count_nonzero(blob)*.9
    crossing=np.zeros_like(blob)
    crossing[76:85,20:140]=True
    crossing[20:140,76:85]=True
    suppressed,protected=suppress_wide_patches(crossing,5)
    assert not suppressed[80,80]
    assert protected[80,80]


def test_segment_tile_invariance_and_region_exclusion(tmp_path):
    image,_,_=synthetic_scene(320)
    preview=image.astype(np.float32)/255
    cfg=Config(tile_size=96,preview_size=320)
    geo=estimate_geometry(preview,image.shape,cfg)
    corrected=np.maximum(preview-geo.background,0)
    ridges=ridge_response(corrected,[max(.7,s*geo.diameter/1000) for s in cfg.ridge_sigmas],cfg.ridge_beta,cfg.ridge_gamma)
    first=np.zeros_like(image);second=np.zeros_like(image)
    region=np.zeros_like(image);candidate=np.zeros_like(image)
    segment(corrected,ridges,geo,cfg,tmp_path,first,candidate,region)
    segment(corrected,ridges,geo,replace(cfg,tile_size=1024),tmp_path,second)
    np.testing.assert_array_equal(first,second)
    assert not np.any(first[region == 0])
    assert not np.any(first[candidate == 0])


def test_failed_geometry_has_diagnostics_and_no_measurement(tmp_path):
    path=tmp_path/"empty.tif"
    tifffile.imwrite(path,np.zeros((128,128),np.uint8))
    out=tmp_path/"failed"
    result=run_image(path,out,Config(),False)
    assert result["status"] == "failed"
    assert result["tunnel_area_px"] is None
    assert (out/"empty.tif_overlay.png").exists()
    assert not (out/"tunnels.tif").exists()
    assert main([str(path),"--output",str(out)]) == 2


def test_end_to_end_synthetic_and_missing_annotations(tmp_path):
    image,truth,_=synthetic_scene(480,noisy=True)
    source=tmp_path/"scene.tif";tifffile.imwrite(source,image)
    before=source.read_bytes()
    out=tmp_path/"result"
    config=Config(preview_size=480,tile_size=512)
    crops=tmp_path/"annotations"
    result=run_image(source,out,config,True,validation_dir=crops)
    truth=synthetic_reference(truth,config)
    assert result["status"] == "provisional"
    mask=audit_mask(out)
    region=audit_mask(out,"analysis_region")
    assert set(p.name for p in out.iterdir()) == {"results.csv","scene.tif_overlay.png","audit.zip"}
    assert result["tunnel_area_px"] == int(mask.sum())
    assert not np.any(mask[region == 0])
    assert source.read_bytes() == before
    metrics=compare_masks(mask,truth)
    assert metrics["precision"] > .85
    assert metrics["recall"] > .7
    assert result["sensitivity"] is not None
    report=validate_crops(crops)
    assert report["status"] == "incomplete"
    assert len(report["missing_reference_masks"]) == 6
    manifest=json.loads((crops/"manifest.json").read_text())
    for crop in manifest["crops"]:
        tifffile.imwrite(crops/crop["expected_reference"],tifffile.imread(crops/crop["prediction"]))
    assert validate_crops(crops)["status"] == "complete"


def test_broad_network_is_retained(tmp_path):
    image,truth,debris=synthetic_scene(640,noisy=True,width_multiplier=3)
    source=tmp_path/"broad.tif";tifffile.imwrite(source,image)
    result=run_image(source,tmp_path/"out",Config(),False)
    assert result["status"] == "provisional"
    mask=audit_mask(tmp_path/"out")
    metrics=compare_masks(mask,synthetic_reference(truth,Config()))
    assert metrics["recall"] >= .95
    assert metrics["precision"] >= .95
    assert abs(metrics["relative_area_error"]) <= .05


def test_geometry_and_segmentation_repeat_deterministically(tmp_path):
    image,_,_=synthetic_scene(256)
    preview=image.astype(np.float32)/255
    config=Config(preview_size=256,tile_size=256)
    first=estimate_geometry(preview,image.shape,config)
    second=estimate_geometry(preview,image.shape,config)
    assert first.describe() == second.describe()
    np.testing.assert_array_equal(first.inner_mask,second.inner_mask)
    np.testing.assert_array_equal(first.outer_cutoffs,second.outer_cutoffs)


def test_cli_folder_continues_after_invalid_input(tmp_path):
    source=tmp_path/"input";source.mkdir()
    tifffile.imwrite(source/"a-stack.tif",np.zeros((4,64,64),np.uint8),photometric="minisblack")
    image,_,_=synthetic_scene(256)
    tifffile.imwrite(source/"b-scene.tif",image)
    output=tmp_path/"out"
    assert main([str(source),"--output",str(output),"--no-sensitivity"]) == 2
    assert (output/"b-scene.tif_overlay.png").exists()
    assert "failed" in (output/"results.csv").read_text()


def test_two_arm_blob_keeps_branch_corridor():
    from scipy import ndimage as ndi
    yy,xx=np.mgrid[:160,:160]
    branch=(abs(yy-80)<=2)&(xx>=10)&(xx<=150)
    blob=(xx-80)**2+(yy-80)**2 <= 14**2
    candidate=branch|blob
    rejected,restored=suppress_wide_patches(candidate,5)
    accepted=candidate & ~rejected
    assert np.all(accepted[80,10:151])
    assert np.count_nonzero(accepted & blob & ~branch) < np.count_nonzero(blob & ~branch)*.3
    assert restored[80,80]


def test_gradually_widening_branch_is_not_cut():
    yy,xx=np.mgrid[:160,:200]
    half_width=3+8*np.maximum(0,1-abs(xx-100)/90)
    branch=(abs(yy-80)<=half_width)&(xx>8)&(xx<192)
    rejected,_=suppress_wide_patches(branch,5)
    assert not np.any(rejected & branch)


@pytest.mark.parametrize("irregular",[False,True])
def test_fixed_inner_buffer_uses_euclidean_distance(irregular):
    from scipy import ndimage as ndi
    image,_,_=synthetic_scene(400)
    config=Config(preview_size=400)
    geometry=estimate_geometry(image.astype(np.float32)/255,image.shape,config)
    yy,xx=np.mgrid[:400,:400]
    opening=(xx-200)**2+(yy-200)**2 < 45**2
    if irregular:
        opening |= (xx-230)**2+(yy-188)**2 < 30**2
    geometry.inner_mask=opening
    region,_=tile_fields(geometry,image.shape,0,400,0,400,config)
    outer=outer_region(xx,yy,geometry.center_x,geometry.center_y,geometry.radius_x,geometry.radius_y,geometry.angle,geometry.outer_cutoffs)
    expected=outer & (ndi.distance_transform_edt(~opening)>geometry.diameter*.025)
    np.testing.assert_array_equal(region,expected)
    assert geometry.details["inner_buffer_px"] == geometry.diameter*.025


def test_ring_only_has_zero_area_and_compact_outputs(tmp_path):
    yy,xx=np.mgrid[:400,:400]
    radius=np.hypot(xx-199.5,yy-199.5)
    rings=((radius>=46)&(radius<=50))|((radius>=184)&(radius<=186))
    path=tmp_path/"rings.tif"
    tifffile.imwrite(path,(rings*255).astype(np.uint8))
    result=run_image(path,tmp_path/"out",Config(),False)
    assert result["status"] == "provisional"
    assert result["tunnel_area_px"] == 0
    assert set(p.name for p in (tmp_path/"out").iterdir()) == {"results.csv","rings.tif_overlay.png","audit.zip"}


def test_audit_and_overlay_use_the_final_mask(tmp_path):
    import zipfile,hashlib,io
    from PIL import Image
    from tunnel_analysis.report import save_overlay,small_mask
    image,_,_=synthetic_scene(256)
    path=tmp_path/"scene.tif";tifffile.imwrite(path,image)
    out=tmp_path/"out"
    debug=tmp_path/"debug"
    result=run_image(path,out,Config(preview_size=256),False,debug_dir=debug)
    with zipfile.ZipFile(out/"audit.zip") as archive:
        record=json.loads(archive.read("run.json"))["results"][0]
        entry=record["masks"]["tunnels"]
        content=archive.read(entry["archive_path"])
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        mask=tifffile.imread(io.BytesIO(content))
    assert np.count_nonzero(mask) == result["tunnel_area_px"]
    layer=save_overlay(tmp_path/"overlay.png",image.astype(np.float32)/255,mask,
        tifffile.imread(debug/"foreground_candidates.tif"),audit_mask(out,"analysis_region"),result)
    np.testing.assert_array_equal(layer,mask>0)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/"overlay.png")),np.asarray(Image.open(out/"scene.tif_overlay.png")))


def test_diagnostics_cannot_be_nested_in_results(tmp_path):
    from tunnel_analysis.pipeline import check_destinations
    with pytest.raises(ValueError,match="non-nested"):
        check_destinations(tmp_path/"out",tmp_path/"out"/"debug")
