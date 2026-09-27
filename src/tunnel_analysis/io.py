"""TIFF input and small display images; no full-size RGB overlays."""
from pathlib import Path
import hashlib
import numpy as np
import tifffile
from PIL import Image


class InputImage:
    def __init__(self, path):
        self.path = Path(path)
        with tifffile.TiffFile(path) as tf:
            if len(tf.series) != 1 or len(tf.pages) != 1:
                raise ValueError("Expected one 2-D TIFF image; stacks and multiple series are unsupported")
            series = tf.series[0]
            shape, dtype = series.shape, series.dtype
            photometric = tf.pages[0].photometric.name
            if not (len(shape) == 2 or (len(shape) == 3 and shape[-1] == 3 and photometric == "RGB")):
                raise ValueError(f"Expected grayscale YX or RGB YXS, got {shape}, axes={series.axes}")
            if dtype.kind != "u" or dtype.itemsize not in (1, 2):
                raise ValueError(f"Only unsigned 8-bit and 16-bit TIFFs are supported, got {dtype}")
            if photometric not in ("MINISBLACK", "MINISWHITE", "RGB"):
                raise ValueError(f"Unsupported TIFF photometric interpretation: {photometric}")
            orientation = tf.pages[0].tags.get("Orientation")
            if orientation and orientation.value != 1:
                raise ValueError("TIFF must use top-left orientation; normalize orientation before analysis")
            self.metadata = {"shape": list(shape), "dtype": str(dtype), "axes": series.axes,
                             "photometric": photometric, "imagej": tf.imagej_metadata,
                             "tiff_resolution_is_specimen_calibration": False}
            for k in ("XResolution", "YResolution", "ResolutionUnit"):
                if k in tf.pages[0].tags:
                    self.metadata[k] = tf.pages[0].tags[k].value
        self.maximum = float(np.iinfo(dtype).max)
        self.invert = photometric == "MINISWHITE"
        try:
            self.array = tifffile.memmap(path, mode="r")
            self.metadata["memory_mapped"] = True
        except ValueError:
            self.array = tifffile.imread(path)
            self.metadata["memory_mapped"] = False
        self.shape = shape[:2]

    def gray(self, ys=slice(None), xs=slice(None)):
        a = self.array[ys, xs].astype(np.float32)
        if a.ndim == 3:
            a = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        a /= self.maximum
        return 1 - a if self.invert else a

    def preview(self, max_size):
        # BOX resampling averages fine lines instead of losing them to stride aliasing.
        step = max(1, int(np.ceil(max(self.shape) / (max_size * 2))))
        small = self.gray(slice(None, None, step), slice(None, None, step))
        h, w = self.shape
        factor = min(1.0, max_size / max(h, w))
        size = (round(w * factor), round(h * factor))
        return np.asarray(Image.fromarray(small).resize(size, Image.Resampling.BOX)).copy()

    def close(self):
        if isinstance(self.array, np.memmap):
            self.array._mmap.close()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png(path, array):
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.dtype.kind == "f":
        array = np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def resize_mask(mask, shape):
    return np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
