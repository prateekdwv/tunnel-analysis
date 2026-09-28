# Tunnel area analysis

Measure visible white tunnel-network area in TIFF images, with one shared automatic workflow. Branches and junctions count, including genuine tunnels near the centre. Debris, supported smooth inner-rim material, and the outer rim are excluded. Dark spaces between tunnel walls remain uncounted.

## Keep developing here; run on Google Drive when ready

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekdwv/tunnel-analysis/blob/main/notebooks/Google_Drive_Analysis.ipynb)

The [Colab notebook](notebooks/Google_Drive_Analysis.ipynb) launches the existing Python package. Keep editing and testing in VS Code Codespaces. When ready, commit and push the code and notebook to GitHub; the button above will work once the notebook is on `main`. The launcher runs the measurement rule in the selected commit. The current rule includes central tunnels; older commits used a buffered central exclusion.

In Colab, choose a CPU runtime, enter the mounted Drive input folder and results parent folder, authorize Drive, and run the cells. The notebook downloads `main` by default, resolves it to an exact Git commit, and installs it in an isolated local Python environment. An optional branch, tag, or commit can select another version. It prints the chosen commit and a unique run directory to keep for resuming.

Each new run processes the TIFFs directly inside the input folder. Results go to a separate dated directory containing the CSV, comparison graph, permanent overlays, and audit ZIP. Editing or pushing code here does not change a running analysis. The notebook is manually started; it does not watch Drive for new uploads.

The launcher and recovery behavior have been tested locally. A user-reported Colab batch completed with three provisional measurements and one failed central-opening detection under the older rule. The revised central rule has not been validated on those four original TIFFs. [Google's documentation](https://research.google.com/colaboratory/faq.html) describes runtime limits and Drive access.

## Read the results

Open **[results/results.csv](results/results.csv)** for the measurements and the **`*_overlay.png`** files beside it to check the segmentation.

| Overlay color | Meaning |
|---|---|
| Green | Accepted tunnel pixels that contribute to area |
| Orange | Foreground pixels rejected from the measurement |
| Blue | Excluded from the tunnel count; rim exclusions remain in the coverage denominator |
| Pale blue boundary | Edge of the region eligible for tunnel segmentation |
| White boundary | Fitted arena used for the coverage denominator |

Each overlay includes the filename, area, percentage coverage, and a legend. It is always saved. The image is a display preview: reduced masks preserve visibility of thin lines, so apparent preview widths must not be used for measuring area.

The legacy image/folder command creates five files for three images (the resumable `batch` command also saves `comparison.png`):

```text
results/
  results.csv
  ctrl 300 dpi.tif_overlay.png
  gal 80 control45-1 cropped.tif_overlay.png
  nsp3 300 dpi.tif_overlay.png
  audit.zip
```

- `tunnel_area_px`: number of accepted native-resolution pixels.
- `arena_area_px`: number of observed native pixels inside the fitted arena, including the centre and rim-artifact locations.
- `analysis_region_area_px`: compatibility alias for `arena_area_px`; its meaning changed in `arena-coverage-v1`.
- `tunnel_area_percent`: `100 × tunnel_area_px / arena_area_px`.
- `status` and `notes`: provisional status and any review flags. A failed measurement has no area value.

**The centre is eligible for measurement.** A missing or irregular inner rim no longer makes a measurement fail. Only a narrow, strongly supported smooth rim is excluded; the old 2.5%-diameter central buffer is removed. Genuine central tunnel pixels use the same foreground/branch checks as the rest of the arena. Dark central pixels add no tunnel area but remain part of the arena denominator. Artifact rejection never shrinks that denominator.

The detector requires support near an ellipse over at least 85% of angular samples and protects radial branch connections. It cannot distinguish every circular tunnel from a rim artifact or identify all irregular remnants. Every result flags either `inner_rim_exclusion_requires_review` or `inner_rim_not_identified_review_centre`; the latter **does not confirm that the rim is absent**. In the three local biological previews, this conservative detector did not confidently identify an inner rim. Review is still needed before making biological comparisons.

This is a changed measurement definition (`arena-coverage-v1` in the CSV and audit), so rerun all comparison images together in a **new run**. Existing results and checkpoints retain their original rule and must not be mixed with new results. Old configurations containing `inner_buffer_fraction` are rejected; use the revised `config.json` or a historical code revision with its original settings. Do not resume an older run to apply the new rule.

`audit.zip` holds exact 0/1 TIFF masks (`tunnels.tif`, `analysis_region.tif` for the full arena denominator, and `segmentation_region.tif` for eligibility after rim exclusions) under `images/<input filename>/`, and a `run.json` with configuration, image/code/mask checksums, software versions, geometry, timing, memory use, and sensitivity results. Extract a mask and set its display range to 0–1 in ImageJ/Fiji if it appears black. Intermediate analysis files are temporary and are not left in the results.

The old development outputs are preserved in [archive/previous-development-results.zip](archive/previous-development-results.zip). The input TIFFs are unchanged.

See [VALIDATION.md](VALIDATION.md) for current checks and clearly separated historical results and [METHODOLOGY.md](METHODOLOGY.md) for the scientific method. **Biological accuracy remains provisional until compared with independent expert annotations.**

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

By default the analysis also repeats classification at 90% and 110% of both intensity and ridge thresholds. Add `--no-sensitivity` to skip these checks. Historical results used the older buffered-centre settings. The current shared settings are in [config.json](config.json).

## Resumable batches, locally or on mounted Drive

The existing commands above remain available. The new batch command adds checkpoints and recovery, using ordinary filesystem paths so it can be tested here:

```bash
tunnel-analysis batch /path/to/photos --output /path/to/Results/run_001 --config config.json
tunnel-analysis batch /path/to/photos --output /path/to/Results/run_001 --resume
```

Folder scanning is deterministic, includes `.tif` and `.tiff` regardless of extension case, and does not recurse. Each image is copied to local scratch storage and analyzed in a fresh worker. Original images are never modified. Specify an existing local `--scratch-dir` to choose working storage; the notebook uses `/content`, not Drive, for processing and intermediate files.

On resume, omitted configuration and sensitivity options use the saved settings. Explicitly different options are rejected. Optional `--source-root /content/drive` records source locations such as `MyDrive/experiment/control.tif` in the audit; locally the default root is the input directory. Both absolute input locations and relative source paths are recorded, so keep folder locations unchanged when resuming.

### Checkpoints and completed results

During a run, verified per-image checkpoints and `state.json` are stored beside the final folder:

```text
Results/
  .tunnel-checkpoints/
    run_001/              # recovery data while the run is incomplete
  run_001/                # final CSV, comparison.png, overlays, and audit.zip
```

The final folder is assembled after all images have been attempted. The audit is uploaded last as the completion record. Checkpoints are removed only after final files have been read back and verified. Do not delete recovery data during an interrupted run, and use only one active process/session per run. Partially published results are repaired from checkpoints on resume.

- Completed images are reused only after source and checkpoint verification. Missing or damaged per-image checkpoints are rebuilt.
- Changed file contents, added/removed TIFFs, different configuration, sensitivity settings, code, Git commit, Python, or dependency versions require a new run, or restoration of the original inputs/environment.
- Source checksums require sequential reads of the original TIFFs at the start and end; staging also verifies the copied image. Resume still reads sources for verification, but valid completed images skip copying and analysis.
- Unsupported images and failed region detection receive failed CSV rows and clearly labeled failure overlays. Remaining images continue. Such a finished batch is marked `complete_with_failures` and returns exit code 2. Corrected inputs require a new run.
- Resuming an intact completed run verifies its artifacts and leaves them unchanged. Damaged completed results without checkpoints are preserved and reported as an error; choose a new run instead of overwriting them.

The audit's `batch` section records completion status, the input list and checksums, configuration, sensitivity setting, source-root location, Git commit and dirty status, code checksum, Python version, and all installed dependency versions. Each image also records its original source path and checksum. A local run with uncommitted code remains resumable in that same unchanged environment, but cannot be reconstructed from GitHub by Colab.

### Resuming in Colab

Paste the complete saved run directory into **RESUME_RUN**, reconnect Drive, and run the notebook again. The notebook reads the saved input folder, fetches the original Git commit, and reinstalls the recorded dependency versions in a fresh environment. It requires the same Python version. If that environment is unavailable, keep the old run and start a new one. New-run folder, revision, and sensitivity fields are ignored when resuming.

For new Colab runs, package dependencies are resolved for the hosted Python version and recorded in the audit. The existing `requirements-lock.txt` describes the local Python 3.14.2 environment and is not applied blindly to Colab. The analysis subprocess is isolated from packages already imported in the notebook kernel.

Integration checks use synthetic images and local directories only:

```bash
pytest -q
```

They cover interruption, checkpoint damage, publication recovery, mismatched inputs/settings, failure continuation, native mask agreement, source preservation, and offline Git branch/tag/commit selection. No biological TIFF rerun, Drive connection, or GitHub push is needed for these tests.

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

The program uses CPU processing, overlapping tiles, and disk-backed intermediates. Temporary storage is approximately 17 bytes per source pixel with sensitivity enabled, plus temporary compressed archives. Allow around 1.5 GB of working disk space for the largest supplied image. Set `TMPDIR` to choose the temporary disk. Final masks compress into `audit.zip`; GPU and trained-model dependencies are not required.

### Batch comparison graph

Each completed batch saves a single-panel `comparison.png`: visible tunnel coverage as a percentage of the fitted arena. The pixel-area panel has been removed. Pixel counts remain in the CSV and audit.

- **Denominator:** every observed native pixel inside the fitted arena ellipse. Inner and outer rim rejection, debris rejection, and tunnel thresholds do not shrink it. The centre is included. The numerator retains the existing segmentation and rim-rejection rules.
- **Presentation:** horizontal blue bars, white background, readable bundled DejaVu Sans fonts, restrained gridlines, direct percentage labels, and a linear axis starting at zero. The upper limit rounds up to a readable tick above the data, capped at 100%; separate runs may use different upper limits, so consult the ticks when comparing charts.
- **Sample labels:** omit TIFF extensions and DPI tokens and replace underscores with spaces. Original filenames are preserved in CSV/audit. Colliding display labels retain identifying filenames. No filename-specific colours or inferred group assignments are used.
- **Missing values:** failed or invalid measurements have a labeled row without a bar; zero is a valid measured value. An estimated missing arena fraction above 0.5% (`arena_missing_fraction_warning`) is flagged, and its observed-area percentage is omitted from the comparison chart pending review. The audit records `arena_estimated_missing_fraction = max(0, 1 − observed arena pixels / fitted ellipse area)`; the 0.5% review tolerance allows minor fit/rasterization discrepancies and is not a biological validation threshold. No missing area is extrapolated as zero tunnels. The arena fit is an image-based approximation of the experimental surface and should be checked against the white overlay outline.
- **Export:** PNG at 300 dpi, 7.1 inches wide (2,130 pixels), with height adapted to sample count and label wrapping. `audit.zip` additionally contains an editable vector `figures/comparison.svg` and `figures/comparison.json` with the caption, label mapping, plotted values, axis limits, and rendering version. The SVG can be scaled for a journal layout without raster blur.
- **Interpretation:** each bar is one image, without biological replicate error bars or significance tests. Measurements remain provisional pending independent validation. Coverage normalizes image area but cannot correct lost fine detail, washing damage, or inconsistent specimen preparation.

The graph uses Matplotlib, imported only during final batch assembly, and never reads source TIFFs or masks. Its PNG and archived SVG checksums are verified on resume. The final output layout remains compact: for four images, seven files (CSV, PNG chart, four overlays, audit ZIP). The notebook displays the PNG and explains where to find the vector version.

Historical CSVs cannot simply be redrawn under the new axis label: their percentages used different denominators. Start a new run with all comparison images to apply `arena-coverage-v1`; do not resume an earlier run for that purpose. Original runs remain untouched.

The notebook defaults to `/content/drive/MyDrive/tunnel-quant/priority` for inputs and `/content/drive/MyDrive/tunnel-quant/Results` for new run folders.

## Revise selected Drive images

To restore the earlier central exclusion for an image with a confirmed opening:

```bash
tunnel-analysis revise /path/to/priority --from-run /path/to/Results/previous_run \
  --output /path/to/Results/revised_run --image "NSP3 300 dpi.tif" \
  --inner-exclusion opening-buffer
```

Repeat `--image` for multiple exact filenames. Only those source TIFFs are read and analyzed. The other measurements, masks and overlays are carried forward unchanged. The combined CSV and graph are rebuilt. The original run and all source images remain unchanged. Selected TIFF checksums must match the parent run; this command corrects analysis, not replacement source images.

`opening-buffer` restores the detected central opening plus a Euclidean buffer of 2.5% of arena diameter, including branch portions inside that band. The overlay shows the excluded region in blue. The full-arena percentage denominator is unchanged. Other segmentation parameters and the parent's sensitivity setting are inherited. Default analysis remains `rim-only`; merely specifying `inner_buffer_fraction` does not enable the opening exclusion. A confirmed washed-away opening should retain `rim-only`.

In Colab, set **REVISE_RUN** to the completed parent folder, **SELECTED_IMAGES** to exact filenames (one per line in the Python string, separated by `\n`), and enable **RESTORE_INNER_OPENING**. Leave **RESUME_RUN** empty initially. The notebook creates a new combined results folder. To resume an interrupted revision, enter that **new folder** in RESUME_RUN; the notebook restores the revision's code, environment, selection and settings.

For the CLI, repeat the revision command with `--resume`. Failed selected images preserve checkpoints and prevent publication of a revised result; the original run stays intact. Missing or corrupt image checkpoints are rebuilt. Do not change settings or code while resuming. Earlier runs whose percentages use the reduced analysis region cannot be combined with full-arena measurements: run a new full batch once to establish a compatible baseline. Revisions require a completed batch with a verified audit.

The audit records the parent audit checksum, selected filenames, per-image configurations, code and dependency identities. Carried records retain their original provenance; the new batch identity describes the revision/assembly environment.
