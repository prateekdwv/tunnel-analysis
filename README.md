# Tunnel area analysis

Measure visible white tunnel-network area in TIFF images, with one shared automatic workflow. Branches and junctions count; debris, the central opening and its surrounding band, and the outer rim are excluded. Dark spaces between tunnel walls remain uncounted.

## Read the results

Open **[results/results.csv](results/results.csv)** for the measurements and the **`*_overlay.png`** files beside it to check the segmentation.

| Overlay color | Meaning |
|---|---|
| Green | Accepted tunnel pixels that contribute to area |
| Orange | Foreground pixels rejected from the measurement |
| Blue | Excluded central region, rim bands, and outside background |
| Pale blue boundary | Edge of the usable analysis region |

Each overlay includes the filename, area, percentage coverage, and a legend. It is always saved. The image is a display preview: reduced masks preserve visibility of thin lines, so apparent preview widths must not be used for measuring area.

A three-image run creates **exactly five files**:

```text
results/
  results.csv
  ctrl 300 dpi.tif_overlay.png
  gal 80 control45-1 cropped.tif_overlay.png
  nsp3 300 dpi.tif_overlay.png
  audit.zip
```

- `tunnel_area_px`: number of accepted native-resolution pixels.
- `analysis_region_area_px`: usable area after the central and outer exclusions.
- `tunnel_area_percent`: `100 × tunnel_area_px / analysis_region_area_px`.
- `status` and `notes`: provisional status and any review flags. A failed measurement has no area value.

**The inner exclusion is the detected opening plus an outward buffer equal to 2.5% of the arena diameter.** This applies to every image, including irregular openings. Branch portions inside that band are excluded; outward branches count from where they leave it.

`audit.zip` holds exact 0/1 TIFF masks (`tunnels.tif` and `analysis_region.tif`) under `images/<input filename>/`, and a `run.json` with configuration, image/code/mask checksums, software versions, geometry, timing, memory use, and sensitivity results. Extract a mask and set its display range to 0–1 in ImageJ/Fiji if it appears black. Intermediate analysis files are temporary and are not left in the results.

The old development outputs are preserved in [archive/previous-development-results.zip](archive/previous-development-results.zip). The input TIFFs are unchanged.

See [VALIDATION.md](VALIDATION.md) for checks on these images and [METHODOLOGY.md](METHODOLOGY.md) for the scientific method. **Biological accuracy remains provisional until compared with independent expert annotations.**

## Install and run

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows activate with `.venv\Scripts\activate`; peak POSIX memory accounting is unavailable there and is recorded as `null`. Exact versions for the tested Python 3.14.2/Linux environment are in `requirements-lock.txt`.

One image:

```bash
tunnel-analysis "ctrl 300 dpi.tif" --output control-results --config config.json
```

All TIFFs directly inside the current folder, processed sequentially:

```bash
tunnel-analysis . --output next-results --config config.json
```

Choose a new output directory for each run. Existing results are not overwritten. `python -m tunnel_analysis` is equivalent to the installed command. Folder mode does not recurse. Each input runs in a fresh worker, which makes peak memory measurements independent across images.

By default the analysis also repeats classification at 90% and 110% of both intensity and ridge thresholds. Add `--no-sensitivity` to skip these checks. All three supplied images were measured with the shared [config.json](config.json).

## Optional diagnostics and expert validation

These options are off by default and require separate locations outside the results folder:

```bash
tunnel-analysis "ctrl 300 dpi.tif" --output review-results --config config.json \
  --debug-dir diagnostics/control --validation-dir annotations/control
```

Debug outputs include intermediate previews and masks. The validation location contains six source crops, separate predictions and overlays, coordinates, and annotation instructions, grouped by input filename. Experts should mark source crops without viewing predictions, save the prescribed `*_reference.tif` masks, then compare them:

```bash
tunnel-analysis validate-crops "annotations/control/ctrl 300 dpi.tif" \
  --references expert-masks --output control-comparison.json
```

Missing annotations are reported. These selected diagnostic crops are not an unbiased sample of whole-image accuracy. Freeze settings before independent validation. You can also compare any two matched binary masks:

```bash
tunnel-analysis compare prediction.tif reference.tif --output comparison.json
```

Run tests and synthetic validation separately from biological results:

```bash
pytest -q
tunnel-analysis synthetic --output synthetic-checks --config config.json
```

Synthetic validation exits unsuccessfully if any scene fails 95% precision, 95% recall, or a maximum absolute area error of 5%. These tests measure generated scenes, not biological accuracy.

## Input and resource requirements

Supported inputs are single-page unsigned 8-bit or 16-bit grayscale or interleaved RGB TIFFs with top-left orientation. Stacks, multiple series, palettes, alpha channels, signed/float images, and other orientations are rejected. Some TIFF compression formats require the optional `imagecodecs` package; these inputs are uncompressed.

Spatial parameters scale with arena diameter. Pixel areas from different resolutions cannot be directly compared as physical areas. TIFF DPI is metadata, not specimen calibration.

The program uses CPU processing, overlapping tiles, and disk-backed intermediates. Temporary storage is approximately 16 bytes per source pixel with sensitivity enabled, plus temporary compressed archives. Allow around 1.5 GB of working disk space for the largest supplied image. Set `TMPDIR` to choose the temporary disk. Final masks compress into `audit.zip`; GPU and trained-model dependencies are not required.
