"""Independent-mask comparisons and reproducible synthetic experiments."""
import json
import io
import zipfile
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import draw
from .config import Config
from .io import png
from .pipeline import run_image


def audit_mask(folder,name="tunnels",image_index=0):
    with zipfile.ZipFile(Path(folder)/"audit.zip") as archive:
        record = json.loads(archive.read("run.json"))["results"][image_index]
        return tifffile.imread(io.BytesIO(archive.read(record["masks"][name]["archive_path"]))) > 0


def synthetic_reference(truth,config):
    """Analytic ground-truth exclusion, independent of predicted geometry."""
    size = truth.shape[0]
    yy,xx = np.mgrid[:size,:size]
    radius = np.hypot(xx-(size-1)/2,yy-(size-1)/2)
    inner_cutoff = size*.115+config.inner_buffer_fraction*(2*size*.465)
    return truth & (radius > inner_cutoff)


def binary_mask(path):
    array = tifffile.imread(path)
    if array.ndim != 2 or not np.all(np.isin(np.unique(array),[0,1,255])):
        raise ValueError(f"Expected a 2-D binary mask (0/1 or 0/255): {path}")
    return array > 0


def compare_masks(prediction,reference):
    if prediction.shape != reference.shape:
        raise ValueError(f"Mask shapes differ: {prediction.shape} and {reference.shape}")
    p,r = prediction.astype(bool),reference.astype(bool)
    tp = int(np.count_nonzero(p&r))
    fp = int(np.count_nonzero(p&~r))
    fn = int(np.count_nonzero(~p&r))
    pa,ra = tp+fp,tp+fn
    return {"true_positive_px":tp,"false_positive_px":fp,"false_negative_px":fn,
            "prediction_area_px":pa,"reference_area_px":ra,
            "precision":tp/pa if pa else (1.0 if not ra else 0.0),
            "recall":tp/ra if ra else (1.0 if not pa else 0.0),
            "dice":2*tp/(pa+ra) if pa+ra else 1.0,
            "iou":tp/(tp+fp+fn) if tp+fp+fn else 1.0,
            "signed_area_error_px":pa-ra,
            "relative_area_error":(pa-ra)/ra if ra else None}


def validate_crops(folder,references=None):
    folder = Path(folder)
    references = Path(references) if references else folder
    manifest = json.loads((folder/"manifest.json").read_text())
    comparisons = []
    missing = []
    for crop in manifest["crops"]:
        reference = references/crop["expected_reference"]
        if not reference.exists():
            missing.append(crop["id"])
            continue
        prediction = binary_mask(folder/crop["prediction"])
        expected_shape = (crop["bbox_xyxy"][3]-crop["bbox_xyxy"][1],crop["bbox_xyxy"][2]-crop["bbox_xyxy"][0])
        if prediction.shape != expected_shape:
            raise ValueError(f"Crop {crop['id']} does not match its manifest bounds")
        comparisons.append({"id":crop["id"],**compare_masks(prediction,binary_mask(reference))})
    return {"source_sha256":manifest["source_sha256"],"comparisons":comparisons,"missing_reference_masks":missing,
            "status":"complete" if not missing else "incomplete",
            "interpretation":"diagnostic crop agreement; not an unbiased whole-image accuracy estimate"}


def synthetic_scene(size=640,seed=7,noisy=True,width_multiplier=1.0):
    """Ground-truth visible walls with loops, branches, debris and bright rims."""
    rng = np.random.default_rng(seed)
    yy,xx = np.mgrid[:size,:size]
    cx=cy=(size-1)/2
    rr = np.hypot(xx-cx,yy-cy)
    arena_radius = size*.465
    hole_radius = size*.115
    truth = np.zeros((size,size),bool)

    def stroke(points,width):
        width *= width_multiplier
        m = np.zeros_like(truth)
        for p,q in zip(points[:-1],points[1:]):
            ys,xs = draw.line(round(p[1]),round(p[0]),round(q[1]),round(q[0]))
            good = (ys>=0)&(ys<size)&(xs>=0)&(xs<size)
            m[ys[good],xs[good]] = True
        return ndi.distance_transform_edt(~m) <= width/2

    for i,theta in enumerate(np.linspace(0,2*np.pi,9,endpoint=False)):
        radii = np.linspace(hole_radius+size*.012,arena_radius-size*.015,120)
        angles = theta+.14*np.sin(radii/size*13+i)
        pts = np.column_stack((cx+radii*np.cos(angles),cy+radii*np.sin(angles)))
        truth |= stroke(pts,size*(.0028+.0008*(i%3)))
        start = pts[55]
        tip = start+size*.13*np.array([np.cos(theta+.7),np.sin(theta+.7)])
        truth |= stroke([start,tip],size*.0035)
    ts = np.linspace(0,2*np.pi,240)
    truth |= stroke(np.column_stack((cx+size*.24*np.cos(ts),cy+size*.24*np.sin(ts))),size*.003)
    # Two bright walls with a dark gap: footprint filling would be incorrect.
    truth |= stroke([(size*.22,size*.33),(size*.31,size*.28)],size*.003)
    truth |= stroke([(size*.22,size*.342),(size*.31,size*.292)],size*.003)
    truth |= stroke([(size*.68,size*.69),(size*.76,size*.73)],size*.003)
    truth &= (rr > hole_radius+size*.01)&(rr < arena_radius-size*.012)
    debris = np.zeros_like(truth)
    for px,py,r in ((.71,.38,.017),(.34,.68,.012),(.6,.79,.008),(.23,.48,.005)):
        debris |= (xx-size*px)**2+(yy-size*py)**2 <= (size*r)**2
    # Attach a compact blob to the loop without changing the true line pixels.
    debris |= (xx-(cx+size*.24))**2+(yy-cy)**2 <= (size*.017)**2
    rim = ((rr >= arena_radius-size*.006)&(rr <= arena_radius)) | ((rr>=hole_radius)&(rr<=hole_radius+size*.009))
    background = np.zeros_like(rr,dtype=float)
    if noisy:
        background = (.025+.08*(xx/size)**2+.04*np.sin(yy/size*5)**2)*(rr<arena_radius)
    image = background+.8*truth+.8*(debris & ~truth)+.85*rim
    image = ndi.gaussian_filter(image,.45)
    if noisy:
        image += rng.normal(0,.01,image.shape)*(rr<arena_radius)
    image[rr < hole_radius] = 0
    image[rr > arena_radius+1] = 0
    return np.round(np.clip(image,0,1)*255).astype(np.uint8),truth,debris & ~truth


def synthetic_validation(output,config=None):
    config = config or Config()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    metrics = []
    for name,noisy,width_multiplier in (("clean",False,1),("uneven_noisy",True,1),("broad_tunnels",True,3)):
        image,truth,debris = synthetic_scene(noisy=noisy,width_multiplier=width_multiplier)
        truth = synthetic_reference(truth,config)
        path = output/f"{name}.tif"
        tifffile.imwrite(path,image,photometric="minisblack")
        tifffile.imwrite(output/f"{name}_reference.tif",truth.astype(np.uint8),photometric="minisblack")
        result = run_image(path,output/name,config=config,sensitivity=False)
        if result["status"] == "failed":
            metrics.append({"case":name,"status":"failed","error":result["error"]})
            continue
        predicted = audit_mask(output/name)
        comparison = compare_masks(predicted,truth)
        metrics.append({"case":name,**comparison,
            "acceptance_passed":comparison["precision"] >= .95 and comparison["recall"] >= .95 and abs(comparison["relative_area_error"]) <= .05,
            "debris_false_positive_px":int(np.count_nonzero(predicted & debris)),
            "debris_total_px":int(np.count_nonzero(debris))})
    report = {"seed":7,"size":640,"cases":metrics,
              "acceptance":{"minimum_precision":.95,"minimum_recall":.95,"maximum_absolute_relative_area_error":.05},
              "interpretation":"synthetic accuracy only; does not establish biological accuracy"}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
