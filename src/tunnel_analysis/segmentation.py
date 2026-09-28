"""Tile-local filters followed by global connected-component reconciliation."""
from pathlib import Path
from collections import deque
import mmap
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from .regions import tile_fields


def release_pages(*arrays):
    """Keep disk-backed intermediates from accumulating in resident memory."""
    for array in arrays:
        if isinstance(array,np.memmap):
            array.flush()
            if hasattr(array._mmap,"madvise") and hasattr(mmap,"MADV_DONTNEED"):
                array._mmap.madvise(mmap.MADV_DONTNEED)


def tiles(shape, size, halo=0):
    h,w = shape
    for y in range(0,h,size):
        for x in range(0,w,size):
            y1,x1 = min(y+size,h),min(x+size,w)
            a,b,c,d = max(0,y-halo),min(h,y1+halo),max(0,x-halo),min(w,x1+halo)
            yield (slice(y,y1),slice(x,x1)), (a,b,c,d), (slice(y-a,y1-a),slice(x-c,x1-c))


def ridge_response(image, sigmas, beta, gamma):
    """2-D bright Frangi response with scale-normalized Gaussian Hessians.

    scipy derivatives have finite support (truncate=4), permitting exact tiling.
    Eigenvalues are evaluated analytically, keeping only one scale in memory.
    """
    best = np.zeros_like(image, dtype=np.float32)
    for sigma in sigmas:
        hxx = ndi.gaussian_filter(image,sigma,order=(0,2),mode="reflect",truncate=4)
        hyy = ndi.gaussian_filter(image,sigma,order=(2,0),mode="reflect",truncate=4)
        hxy = ndi.gaussian_filter(image,sigma,order=(1,1),mode="reflect",truncate=4)
        delta = np.sqrt((hxx-hyy)**2+4*hxy*hxy)
        l1 = (hxx+hyy+delta)*(.5*sigma*sigma)
        l2 = (hxx+hyy-delta)*(.5*sigma*sigma)
        swap = np.abs(l1) > np.abs(l2)
        small = np.where(swap,l2,l1)
        large = np.where(swap,l1,l2)
        ratio = np.abs(small)/(np.abs(large)+1e-12)
        value = np.exp(-ratio**2/(2*beta**2)) * (-np.expm1(-(small**2+large**2)/(2*gamma**2)))
        value[large >= 0] = 0
        np.maximum(best,value,out=best)
    return best


def feature_maps(image, geometry, config, scratch, progress=None):
    shape = image.shape
    corrected = np.memmap(Path(scratch)/"corrected.dat",mode="w+",dtype=np.float32,shape=shape)
    ridges = np.memmap(Path(scratch)/"ridges.dat",mode="w+",dtype=np.float32,shape=shape)
    scale = geometry.diameter/1000
    sigmas = [max(.7,s*scale) for s in config.ridge_sigmas]
    halo = int(np.ceil(4*max(sigmas)))+2
    for i,(out,bounds,crop) in enumerate(tiles(shape,config.tile_size,halo)):
        a,b,c,d = bounds
        _,background = tile_fields(geometry,shape,a,b,c,d,config,background_only=True)
        gray = image.gray(slice(a,b),slice(c,d))
        local = np.maximum(gray-background,0)
        corrected[out] = local[crop]
        ridges[out] = ridge_response(local,sigmas,config.ridge_beta,config.ridge_gamma)[crop]
        if progress and i%8 == 0:
            progress(f"Filtering tile {i+1}")
        if out[1].stop == shape[1]:
            release_pages(corrected,ridges,image.array)
    corrected.flush()
    ridges.flush()
    release_pages(corrected,ridges)
    return corrected,ridges


def connecting_skeleton(skeleton, anchors):
    """Prune unanchored dead ends while retaining paths between branch ports."""
    path = skeleton.copy()
    labels,_ = ndi.label(path,structure=np.ones((3,3)))
    attached = np.unique(labels[anchors & path])
    attached = attached[attached > 0]
    path &= np.isin(labels,attached)
    degrees = ndi.convolve(path.astype(np.int16),np.ones((3,3),np.int16))-path
    queue = deque(zip(*np.nonzero(path & ~anchors & (degrees <= 1))))
    height,width = path.shape
    while queue:
        y,x = queue.popleft()
        if not path[y,x] or anchors[y,x] or degrees[y,x] > 1:
            continue
        path[y,x] = False
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            ny,nx = y+dy,x+dx
            if 0 <= ny < height and 0 <= nx < width and path[ny,nx]:
                degrees[ny,nx] -= 1
                if degrees[ny,nx] <= 1 and not anchors[ny,nx]:
                    queue.append((ny,nx))
    return path


def suppress_wide_patches(candidate, half_width):
    """Inspect localized expansions and retain corridors between incoming arms.

    Width is an inspection trigger. Extended ridges remain subject to ridge
    support and the global component filter instead of being cut into pieces.
    Distances are corrected from background-pixel centers to pixel boundaries.
    """
    distance = np.maximum(ndi.distance_transform_edt(candidate)-.5,0)
    cores = distance > half_width
    if not np.any(cores):
        return np.zeros_like(candidate),np.zeros_like(candidate)
    protected = np.zeros_like(candidate)
    inspected = np.zeros_like(candidate)
    labels,n = ndi.label(cores)
    for component_id,sl in enumerate(ndi.find_objects(labels),start=1):
        if sl is None:
            continue
        # A long or gradually widening ridge is not a localized blob. Do not
        # erase it merely because its connected width core is long.
        if max(s.stop-s.start for s in sl) > 4*half_width:
            continue
        pad = int(np.ceil(4*half_width))
        ys = slice(max(0,sl[0].start-pad),min(candidate.shape[0],sl[0].stop+pad))
        xs = slice(max(0,sl[1].start-pad),min(candidate.shape[1],sl[1].stop+pad))
        patch = candidate[ys,xs]
        own_core = labels[ys,xs] == component_id
        blob = (ndi.distance_transform_edt(~own_core) <= half_width+.5) & patch
        py,px = np.nonzero(blob)
        if len(px) >= 3:
            covariance = np.cov(np.vstack([py,px]))
            eigenvalues = np.maximum(0,np.linalg.eigvalsh(covariance))
            if np.sqrt((eigenvalues[1]+.25)/(eigenvalues[0]+.25)) >= 3.0:
                protected[ys,xs] |= blob
                continue
        inspected[ys,xs] |= blob
        # Arms must touch the blob and extend out by at least 1.5 half-widths.
        outside_distance = ndi.distance_transform_edt(~blob)
        skeleton = skeletonize(patch)
        arms = skeleton & ~blob & (outside_distance <= 3*half_width)
        arm_labels,arm_n = ndi.label(arms,structure=np.ones((3,3)))
        near = np.unique(arm_labels[(outside_distance <= 2) & (arm_labels > 0)])
        far = np.unique(arm_labels[(outside_distance >= 1.5*half_width) & (arm_labels > 0)])
        entering = np.intersect1d(near,far)
        if len(entering) >= 2:
            anchors = arms & np.isin(arm_labels,entering)
            path = connecting_skeleton(skeleton & (blob|anchors),anchors)
            if not np.any(path & blob):
                continue
            incoming_widths = []
            for arm in entering:
                samples = (arm_labels == arm) & (outside_distance >= half_width)
                if np.any(samples):
                    incoming_widths.append(float(np.median(distance[ys,xs][samples])))
            if not incoming_widths:
                continue
            # A junction permits the meeting of differently sized arms. A
            # two-arm corridor tracks the neighboring branch widths closely.
            allowance = 1.45 if len(entering) >= 3 else 1.15
            radius = max(.75,float(np.percentile(incoming_widths,75))*allowance)+.5
            corridor = ndi.distance_transform_edt(~path) <= radius
            protected[ys,xs] |= blob & corridor
    return inspected & ~protected,protected


def reconcile_components(mask, diameter, config, scratch):
    """Global 8-connected labels and streamed moments; crosses tile seams."""
    shape = mask.shape
    labels = np.memmap(Path(scratch)/"labels.dat",mode="w+",dtype=np.int32,shape=shape)
    count = ndi.label(mask,structure=np.ones((3,3)),output=labels)
    moments = np.zeros((6,count+1),dtype=np.float64)
    for y in range(0,shape[0],128):
        block = np.asarray(labels[y:y+128])
        yy,xx = np.nonzero(block)
        ids = block[yy,xx]
        yf = (yy+y).astype(np.float64)
        xf = xx.astype(np.float64)
        for k,weights in enumerate((None,xf,yf,xf*xf,yf*yf,xf*yf)):
            moments[k] += np.bincount(ids,weights=weights,minlength=count+1)
    areas = moments[0]
    denom = np.maximum(areas,1)
    vx = np.maximum(0,moments[3]/denom-(moments[1]/denom)**2)
    vy = np.maximum(0,moments[4]/denom-(moments[2]/denom)**2)
    cov = moments[5]/denom-moments[1]*moments[2]/denom**2
    delta = np.sqrt((vx-vy)**2+4*cov**2)
    major = np.maximum(0,(vx+vy+delta)/2)
    minor = np.maximum(0,(vx+vy-delta)/2)
    elongation = np.sqrt((major+.25)/(minor+.25))
    extent = 4*np.sqrt(major)
    # Sparse branching networks have low occupancy of their moment ellipse.
    occupancy = areas/(4*np.pi*np.sqrt((major+.25)*(minor+.25)))
    scale = diameter/1000
    keep = (areas >= max(2,config.min_component_area*scale*scale)) & (
        (elongation >= config.min_elongation) |
        ((extent >= config.min_network_extent*scale) & (occupancy < .45)))
    keep[0] = False
    for y in range(0,shape[0],256):
        mask[y:y+256] = keep[labels[y:y+256]]
    labels._mmap.close()
    Path(scratch,"labels.dat").unlink()
    return {"components_before_filter": int(count),"components_kept": int(np.count_nonzero(keep))}


def segment(corrected, ridges, geometry, config, scratch, mask, candidate_output=None,
            region_output=None, factor=1.0, source=None):
    scale = geometry.diameter/1000
    recovery = max(1,config.recovery_radius*scale)
    width = max(2,config.max_half_width*scale)
    halo = int(np.ceil(max(12*width+recovery,8)))+4
    for out,bounds,crop in tiles(corrected.shape,config.tile_size,halo):
        a,b,c,d = bounds
        native_gray = source.gray(slice(a,b),slice(c,d)) if source is not None else None
        region,background = tile_fields(geometry,corrected.shape,a,b,c,d,config,native_gray)
        if native_gray is None:
            native_gray = corrected[a:b,c:d]+background
        candidate = (corrected[a:b,c:d] >= geometry.threshold*factor) & region
        seeds = (ridges[a:b,c:d] >= config.ridge_threshold*factor) & candidate
        if np.any(seeds):
            supported = ndi.distance_transform_edt(~seeds) <= recovery
            # Background opening can turn a solid bright blob into a hollow
            # residual. Use its original bright interior for shape decisions.
            shape_foreground = candidate | ((native_gray >= config.blob_intensity) & region)
            suppressed,protected = suppress_wide_patches(shape_foreground,width)
            accepted = candidate & (supported|protected) & ~suppressed
        else:
            accepted = np.zeros_like(candidate)
        mask[out] = accepted[crop]
        if candidate_output is not None:
            candidate_output[out] = candidate[crop]
        if region_output is not None:
            region_output[out] = region[crop]
        if out[1].stop == corrected.shape[1]:
            release_pages(corrected,ridges,mask,candidate_output,region_output)
    release_pages(corrected,ridges,mask,candidate_output,region_output)
    return reconcile_components(mask,geometry.diameter,config,scratch)


def counts(mask,region,candidates):
    area = int(np.count_nonzero(mask))
    usable = int(np.count_nonzero(region))
    foreground = int(np.count_nonzero(candidates))
    return {"tunnel_area_px":area,"analysis_region_area_px":usable,
            "tunnel_area_percent":100*area/usable if usable else None,
            "candidate_area_px":foreground,
            "rejected_fraction":1-area/foreground if foreground else 0.0}
