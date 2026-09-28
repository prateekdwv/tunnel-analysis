"""One image per worker with compact, transactional public results."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
import shutil
import sys
import tempfile
import time
import zipfile
import numpy as np
import tifffile
from . import __version__
from .config import Config
from .io import InputImage, sha256
from .regions import estimate_geometry, RegionError
from .segmentation import feature_maps, segment, counts, release_pages
from .report import save_previews, export_crops, write_contact_sheet, save_overlay


def peak_memory_mib():
    if resource is None:
        return None
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum/(1024*1024 if sys.platform == "darwin" else 1024)


def overlay_name(path):
    # Include the extension to distinguish name.tif from name.tiff in batches.
    return Path(path).name+"_overlay.png"


def check_destinations(output, debug_dir=None, validation_dir=None):
    paths = [Path(p).resolve() for p in (output,debug_dir,validation_dir) if p is not None]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}; choose a new directory")
    for i,path in enumerate(paths):
        for other in paths[i+1:]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("Results, debug, and validation directories must be separate, non-nested locations")


def manifest(results):
    return {"format_version":3,"measurement_rule_version":"central-tunnels-v1","measurement":"visible tunnel foreground throughout the arena, excluding supported inner rim material and outer rim band",
            "results":results}


def run_image(path, output, config=None, sensitivity=True, progress=None,
              debug_dir=None, validation_dir=None):
    config = config or Config()
    output = Path(output)
    check_destinations(output,debug_dir,validation_dir)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-",dir=output.parent))
    start = time.perf_counter()
    image = None
    maps = []
    try:
        result = {"input":str(Path(path).resolve()),"input_sha256":sha256(path),
                  "started_utc":datetime.now(timezone.utc).isoformat(),
                  "source_code_sha256":hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest(),
                  "status":"provisional","biological_accuracy":"not independently validated",
                  "config":config.to_dict(),
                  "versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-image","tifffile","pillow")},
                  "software_version":__version__,"python":platform.python_version(),"platform":platform.platform(),
                  "quality_flags":[],"area_unit":"pixel","sensitivity":None,
                  "overlay":overlay_name(path)}
        image = InputImage(path)
        result["image"] = image.metadata
        preview = image.preview(config.preview_size)
        audit = zipfile.ZipFile(staging/"audit.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1)
        try:
            try:
                geometry = estimate_geometry(preview,image.shape,config,image)
            except RegionError as exc:
                result.update(status="failed",error=str(exc),quality_flags=["region_detection_failed"],tunnel_area_px=None)
                zero = np.zeros(preview.shape,np.uint8)
                save_overlay(staging/result["overlay"],preview,zero,zero,np.ones_like(zero),result)
            else:
                result["geometry"] = geometry.describe()
                result["quality_flags"].extend(geometry.flags)
                scale = geometry.diameter/1000
                result["effective_parameters"] = {
                    "scale_px_per_1000_diameter":scale,
                    "ridge_sigmas_px":[max(.7,s*scale) for s in config.ridge_sigmas],
                    "recovery_radius_px":max(1,config.recovery_radius*scale),
                    "width_inspection_radius_px":max(2,config.max_half_width*scale),
                    "inner_rim_max_width_px":config.inner_rim_max_width*geometry.diameter,
                    "min_component_area_px":max(2,config.min_component_area*scale*scale),
                    "intensity_threshold":geometry.threshold,"ridge_threshold":config.ridge_threshold}
                with tempfile.TemporaryDirectory(prefix="tunnel-features-") as scratch:
                    scratch = Path(scratch)
                    corrected,ridges = feature_maps(image,geometry,config,scratch,progress)
                    maps.extend([corrected,ridges])
                    masks = {}
                    for name in ("analysis_region","foreground_candidates","tunnels"):
                        masks[name] = tifffile.memmap(scratch/f"{name}.tif",shape=image.shape,dtype=np.uint8,photometric="minisblack")
                        maps.append(masks[name])
                    if progress:
                        progress("Preserving branch connections and rejecting unsupported patches")
                    result["components"] = segment(corrected,ridges,geometry,config,scratch,masks["tunnels"],
                        masks["foreground_candidates"],masks["analysis_region"],source=image)
                    result.update(counts(masks["tunnels"],masks["analysis_region"],masks["foreground_candidates"]))
                    if result["rejected_fraction"] > config.rejection_warning:
                        result["quality_flags"].append("large_foreground_rejection")
                    if result["tunnel_area_px"] == 0:
                        result["quality_flags"].append("no_tunnels_detected")
                    if sensitivity:
                        if progress:
                            progress("Checking joint threshold sensitivity (-10%, +10%)")
                        trial = np.memmap(scratch/"trial.dat",mode="w+",dtype=np.uint8,shape=image.shape)
                        maps.append(trial)
                        areas = {}
                        for factor in (.9,1.1):
                            segment(corrected,ridges,geometry,config,scratch,trial,factor=factor,source=image)
                            areas[str(factor)] = int(np.count_nonzero(trial))
                        baseline = result["tunnel_area_px"]
                        change = max(abs(n-baseline)/max(1,baseline) for n in areas.values())
                        result["sensitivity"] = {"joint_threshold_factors":areas,"max_relative_area_change":change,
                                                 "interpretation":"parameter sensitivity, not a confidence interval"}
                        if change > config.sensitivity_warning:
                            result["quality_flags"].append("threshold_sensitive")
                    if progress:
                        progress("Saving overlay and compressed audit masks")
                    save_overlay(staging/result["overlay"],preview,masks["tunnels"],masks["foreground_candidates"],masks["analysis_region"],result)
                    if debug_dir:
                        debug = Path(debug_dir)
                        debug.mkdir(parents=True)
                        save_previews(debug,image,preview,geometry,config,corrected,ridges,masks["tunnels"],masks["foreground_candidates"],masks["analysis_region"])
                        write_contact_sheet(debug)
                        for name,array in masks.items():
                            array.flush()
                            shutil.copyfile(scratch/f"{name}.tif",debug/f"{name}.tif")
                    if validation_dir:
                        export_crops(validation_dir,image,preview,geometry,masks["tunnels"],masks["foreground_candidates"],masks["analysis_region"],result["input_sha256"],direct=True)
                    result["masks"] = {}
                    for name in ("tunnels","analysis_region"):
                        masks[name].flush()
                        source = scratch/f"{name}.tif"
                        archive_path = f"images/{Path(path).name}/{name}.tif"
                        audit.write(source,archive_path)
                        result["masks"][name] = {"archive_path":archive_path,"sha256":sha256(source)}
                    for array in maps:
                        array._mmap.close()
                    maps.clear()
            result["runtime_seconds"] = time.perf_counter()-start
            result["peak_memory_mib"] = peak_memory_mib()
            result["memory_measurement"] = "process peak RSS; CLI isolates each image in a fresh worker"
            audit.writestr("run.json",json.dumps(manifest([result]),indent=2)+"\n")
        finally:
            audit.close()
        write_summary([result],staging/"results.csv")
        staging.rename(output)
        return result
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    finally:
        for array in maps:
            array._mmap.close()
        if image is not None:
            image.close()


def write_summary(results,path):
    fields = ["image","tunnel_area_px","analysis_region_area_px","tunnel_area_percent","status","notes"]
    with open(path,"w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for result in results:
            notes = result.get("error") or "; ".join(result.get("quality_flags",[])) or "Awaiting independent biological validation"
            writer.writerow({"image":Path(result["input"]).name,
                **{key:result.get(key) for key in fields[1:-1]},"notes":notes})
