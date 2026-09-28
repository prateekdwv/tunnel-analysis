"""Review artifacts and provenance. Colors never enter measurements."""
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
from .io import png, resize_mask


COLORS = {"accepted": (40,220,80),"rejected": (255,155,45),"excluded": (70,130,255)}


def save_overlay(path,preview,mask,candidates,region,result):
    """Permanent labeled display preview; all colors derive from final masks."""
    shape = preview.shape
    accepted = small_mask(mask,shape)
    foreground = small_mask(candidates,shape)
    usable = resize_mask(region,shape)
    rgb = overlay(preview,accepted,foreground,usable)
    boundary = usable ^ ndi.binary_erosion(usable)
    rgb[boundary] = (160,190,255)
    width = max(900,shape[1])
    canvas = Image.new("RGB",(width,shape[0]+126),(20,23,30))
    canvas.paste(Image.fromarray(rgb),((width-shape[1])//2,126))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf",20)
        small = ImageFont.truetype("DejaVuSans.ttf",16)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((16,8),Path(result["input"]).name,fill="white",font=font)
    if result["status"] == "failed":
        label = "FAILED — no area measured: "+result.get("error","processing failed")
    else:
        label = f"Counted area: {result['tunnel_area_px']:,} pixels | {result['tunnel_area_percent']:.3f}% of usable region | Provisional"
    draw.text((16,36),label,fill="white",font=small)
    x = 16
    for title,key in (("Counted","accepted"),("Rejected","rejected"),("Excluded","excluded")):
        draw.rectangle((x,64,x+16,80),fill=COLORS[key])
        draw.text((x+24,61),title,fill="white",font=small)
        x += 170
    draw.text((16,94),"Display preview; exact masks in audit.zip. Centre included; supported rims excluded. Review centre.",
              fill=(205,210,220),font=small)
    canvas.save(path)
    return accepted


def overlay(gray, mask, candidates, region):
    rgb = np.repeat(np.clip(gray*255,0,255).astype(np.uint8)[...,None],3,axis=2)
    for selected,key,opacity in ((~region,"excluded",.35),(candidates & ~mask,"rejected",.7),(mask,"accepted",.7)):
        rgb[selected] = ((1-opacity)*rgb[selected]+opacity*np.array(COLORS[key])).astype(np.uint8)
    return rgb


def small_float(array, shape):
    # A strided view avoids copying the full native float map into Pillow.
    step = max(1,int(np.ceil(max(array.shape)/(2*max(shape)))))
    small = np.asarray(array[::step,::step]).copy()
    return np.asarray(Image.fromarray(small).resize((shape[1],shape[0]),Image.Resampling.BOX))


def small_mask(array,shape):
    # Maximum occupancy makes thin accepted/rejected pixels visible in reviews.
    # Exact native TIFFs and crop overlays remain the measurement source.
    step = max(1,int(np.floor(min(array.shape[0]/shape[0],array.shape[1]/shape[1]))))
    if step == 1:
        return resize_mask(array,shape)
    pooled = np.zeros(((array.shape[0]+step-1)//step,(array.shape[1]+step-1)//step),np.uint8)
    for y in range(pooled.shape[0]):
        row = np.asarray(array[y*step:min((y+1)*step,array.shape[0])])
        reduced = np.max(row,axis=0)
        pooled[y] = np.maximum.reduceat(reduced,np.arange(0,array.shape[1],step))
    return resize_mask(pooled,shape)


def save_previews(out,image,preview,geometry,config,corrected,ridges,mask,candidates,region):
    out = Path(out)
    shape = preview.shape
    sm = small_mask(mask,shape)
    sc = small_mask(candidates,shape)
    sr = resize_mask(region,shape)
    png(out/"01_input.png",preview)
    boundary = np.repeat((preview*255).astype(np.uint8)[...,None],3,axis=2)
    edge = sr ^ ndi.binary_erosion(sr)
    boundary[ndi.binary_dilation(edge)] = (255,220,0)
    png(out/"02_region_boundaries.png",boundary)
    png(out/"03_background.png",geometry.background)
    png(out/"04_background_corrected.png",small_float(corrected,shape))
    png(out/"05_ridge_response.png",small_float(ridges,shape))
    png(out/"06_foreground_candidates.png",sc)
    png(out/"07_final_tunnels.png",sm)
    png(out/"08_overlay.png",overlay(preview,sm,sc,sr))
    (out/"REVIEW.txt").write_text(
        "Overlay: GREEN = accepted tunnel pixels; RED = rejected foreground; BLUE = excluded region.\n"
        "Yellow lines mark the analysis-region boundary. Preview masks use maximum occupancy to keep thin lines visible.\n"
        "Inspect native TIFF masks and native crop overlays for exact boundaries and area.\n"
        "Area counts white network pixels only. Dark gaps between walls are not filled.\n"
        "All results are provisional pending independent expert validation.\n")
    return sm,sc,sr


def export_crops(out,image,preview,geometry,mask,candidates,region,source_hash,direct=False):
    out = Path(out) if direct else Path(out)/"validation_crops"
    out.mkdir(parents=True)
    shape = preview.shape
    sm,sc,sr = small_mask(mask,shape),small_mask(candidates,shape),resize_mask(region,shape)
    scale = shape[0]/image.shape[0]
    size = min(1024,max(128,round(geometry.diameter*.12)))
    radius = max(2,round(size*scale/2))
    kernel = 2*radius+1
    density = ndi.uniform_filter(sm.astype(np.float32),kernel)
    rejection = ndi.uniform_filter((sc & ~sm & sr).astype(np.float32),kernel)
    from skimage.morphology import skeletonize
    skeleton = skeletonize(sm)
    degree = ndi.convolve(skeleton.astype(np.uint8),np.ones((3,3),np.uint8))-skeleton
    junctions = ndi.uniform_filter((skeleton & (degree >= 3)).astype(np.float32),kernel)
    usable = ndi.uniform_filter(sr.astype(np.float32),kernel) > .95
    yy,xx = np.mgrid[:shape[0],:shape[1]]
    cy,cx = geometry.center_y*scale,geometry.center_x*scale
    radius_map = np.hypot(xx-cx,yy-cy)
    boundary = sr ^ ndi.binary_erosion(sr)
    scores = {
        "tunnels": np.where(usable & (density > .01),1-np.abs(density-.08),-1),
        "junctions": np.where(usable,junctions,-1),
        "debris": np.where(usable,rejection,-1),
        "inner_rim": (np.where(boundary & (radius_map < geometry.diameter*scale*.32),sc.astype(float)+preview,-1)
                      if geometry.details["inner_rim_model"] else -radius_map),
        "outer_rim": np.where(boundary & (radius_map > geometry.diameter*scale*.4),sc.astype(float)+preview,-1),
        "background": np.where(usable,1-density-rejection,-1),
    }
    manifest = {"source_sha256":source_hash,"source_shape":list(image.shape),
                "annotation_status":"not_annotated","selection":"fixed deterministic diagnostic crops; not a random accuracy sample",
                "crops":[]}
    for name,score in scores.items():
        # Keep each crop inside the image and separate from earlier selections where possible.
        score[:radius,:] = -1
        score[-radius:,:] = -1
        score[:,:radius] = -1
        score[:,-radius:] = -1
        py,px = np.unravel_index(np.argmax(score),shape)
        y0 = int(np.clip(round(py/scale)-size//2,0,max(0,image.shape[0]-size)))
        x0 = int(np.clip(round(px/scale)-size//2,0,max(0,image.shape[1]-size)))
        y1,x1 = min(image.shape[0],y0+size),min(image.shape[1],x0+size)
        sl = (slice(y0,y1),slice(x0,x1))
        gray = image.gray(*sl)
        png(out/f"{name}_image.png",gray)
        tifffile.imwrite(out/f"{name}_prediction.tif",np.asarray(mask[sl],dtype=np.uint8),photometric="minisblack")
        png(out/f"{name}_overlay.png",overlay(gray,np.asarray(mask[sl],bool),np.asarray(candidates[sl],bool),np.asarray(region[sl],bool)))
        manifest["crops"].append({"id":name,"bbox_xyxy":[x0,y0,x1,y1],
            "image":f"{name}_image.png","prediction":f"{name}_prediction.tif",
            "expected_reference":f"{name}_reference.tif"})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"ANNOTATION.md").write_text(
        "# Independent expert annotation\n\n"
        "Annotate each `*_image.png` at its original size, without viewing the prediction. "
        "Save a single-channel binary TIFF as `*_reference.tif` (0/1 or 0/255). "
        "Count visible tunnel walls and junctions; exclude dark spaces, debris, and smooth arena rims.\n\n"
        "Predictions are supplied separately for later comparison. "
        "Do not use them to create reference masks. Record annotator, date, uncertain regions, "
        "and any adjudication in a separate annotation log. These deliberately selected crops "
        "test difficult cases and do not provide an unbiased whole-image accuracy estimate.\n\n"
        "Freeze the configuration before evaluation. Use `tunnel-analysis validate-crops "
        "PATH_TO_THIS_FOLDER --output comparison.json`. For independent masks elsewhere, "
        "add `--references PATH_TO_REFERENCES`. Missing annotations are reported, never treated as background.\n")


def write_contact_sheet(out):
    out = Path(out)
    names = ["01_input.png","02_region_boundaries.png","04_background_corrected.png", "05_ridge_response.png","06_foreground_candidates.png","08_overlay.png"]
    sheet = Image.new("RGB",(1200,1260),"#171717")
    draw = ImageDraw.Draw(sheet)
    for i,name in enumerate(names):
        im = Image.open(out/name).convert("RGB")
        im.thumbnail((590,390))
        x,y = (i%2)*600,(i//2)*420
        sheet.paste(im,(x+(600-im.width)//2,y+25))
        draw.text((x+10,y+5),name,fill="white")
    sheet.save(out/"00_contact_sheet.jpg",quality=90)
