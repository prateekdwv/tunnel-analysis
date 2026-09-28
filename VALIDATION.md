# Validation report: central-tunnels-v1

## Current implementation checks

The revised rule includes the central interior, removes the blanket 2.5%-diameter buffer, and permits analysis when no supported inner rim is identified. A narrow smooth rim can be excluded only with strong angular foreground support; radial connections are protected. Both detected and unidentified rims require review. Outer geometry failure still fails the measurement.

**49 tests passed in 16.16 seconds** in the local environment. Checks cover centre-crossing/central tunnels with absent and fragmented rims, an irregular central tunnel loop, a surviving rim with adjoining branches, a ring-only image with zero tunnel area, diameter scaling, native tile agreement, exact counts, determinism, unchanged inputs, overlays, batch recovery, and graph integrity. Two superseded buffer-shape tests were replaced with tests of the revised measurement definition; existing tunnel accuracy requirements were retained. Upstream SciPy/scikit-image deprecation warnings remain.

The three existing synthetic scenes were rerun with all known tunnel pixels in the reference (including pixels previously removed by the central buffer). The results satisfy precision and recall >=95% and absolute relative area error <=5%. Full counts are in [central_tunnels_metrics.json](validation/central_tunnels_metrics.json).

| Scene | Precision | Recall | Signed area error |
|---|---:|---:|---:|
| Clean | 96.82% | 99.51% | +2.774% |
| Uneven illumination and noise | 96.58% | 99.51% | +3.032% |
| Broad tunnels | 97.68% | 98.98% | +1.335% |

These figures are synthetic segmentation checks, not evidence of biological accuracy. Shape alone cannot establish whether a smooth central curve is a rim or a tunnel. The detector deliberately leaves unsupported or irregular structures eligible for segmentation; this may retain unwanted rim remnants.

## Biological validation still outstanding

Preview-only geometry checks on the three local TIFFs (control, gal 80, and nsp3) produced `inner_rim_not_identified_review_centre` in all three. Thus the revised detector does **not** yet establish reliable removal of their biological rim material. No revised native-area measurement or segmentation overlay review was performed on them. Existing biological results were preserved.

The user confirmed that the central rim was washed away in `NSP3, STAT RNAi.tif` and the remaining white lines are genuine tunnels. Only its failed display preview is available in this workspace; its original TIFF and the fourth image from the Drive batch are not available here. The new code removes the specific mandatory-opening failure, but segmentation of that original image has not been tested. It must remain provisional until reviewed.

Before comparing samples, run all four together in a new run using the frozen revised configuration, inspect central accepted/rejected pixels and any rim exclusion, and record threshold sensitivity and runtime/memory from those new audits. An old completed run cannot be resumed to apply a changed measurement definition. Historical coverage values below use a different denominator and must not be mixed with revised values.

---

# Historical validation: buffered-centre rule (0.2.0)

Validated on 27 September 2026 using tunnel-analysis 0.2.0. All three supplied TIFFs were processed with the same historical frozen configuration (stored in the original audits), including an inner buffer of **2.5% of the fitted arena diameter**. Biological measurements remain provisional pending independent expert masks.

## Implementation and synthetic checks

**All 19 automated tests passed.** Checks cover continuous and curved branches, broad and gradually widening branches, junction continuity, isolated and attached debris, the Euclidean buffer around circular and irregular openings, and zero measured area for an excluded inner-ring-only scene. They also check deterministic results, tiled processing, exact pixel counting, source preservation, failed inputs, optional diagnostics, and agreement between the overlay's accepted layer and archived masks.

The three generated scenes passed the required precision and recall of at least 95% and absolute relative area error of at most 5%. Reference exclusions are independently defined from known synthetic geometry, rather than copied from the detected analysis region. Results below use seed 7 and 640 × 640 images; full counts are in [synthetic_metrics.json](validation/synthetic_metrics.json).

| Scene | Precision | Recall | Dice | Signed area error |
|---|---:|---:|---:|---:|
| Clean | 98.88% | 99.50% | 99.19% | +0.625% |
| Uneven illumination and noise | 98.87% | 99.50% | 99.18% | +0.639% |
| Broad tunnels | 99.60% | 98.96% | 99.28% | −0.639% |

Area error is `(predicted area − reference area) / reference area`. These generated examples establish implementation behavior under their simulated conditions; they do not establish accuracy on biological images.

## Supplied images

The final [results.csv](results/results.csv) contains the following measurements. Each overlay is permanently retained beside it, with green for counted pixels, orange for rejected foreground, and blue for excluded regions. Overlays are display previews; native masks in [audit.zip](results/audit.zip) determine every reported area.

| Input / overlay | Tunnel area (pixels) | Analysis region (pixels) | Coverage | Status |
|---|---:|---:|---:|---|
| [ctrl 300 dpi.tif](<results/ctrl 300 dpi.tif_overlay.png>) | 2,565,790 | 16,890,223 | 15.1910% | Provisional |
| [gal 80 control45-1 cropped.tif](<results/gal 80 control45-1 cropped.tif_overlay.png>) | 716,212 | 3,292,452 | 21.7531% | Provisional |
| [nsp3 300 dpi.tif](<results/nsp3 300 dpi.tif_overlay.png>) | 2,551,279 | 42,253,446 | 6.0380% | Provisional |

Pixel counts from different image resolutions are not directly comparable as physical areas. Coverage uses the same region definition across samples, but segmentation accuracy and biological interpretation still require validation.

### Visual review

- **Control:** the previously visible circular rejection gaps along branches are substantially reduced. The expanded blue exclusion removes inner circumferential material and branch bases. Compact masses near junctions remain potentially ambiguous.
- **RGB control:** broad branches are largely preserved. Some compact foreground remains rejected. The run flags `outer_rim_width_at_limit` and `central_boundary_required_gap_closing`, indicating that the outer-rim search reached its configured limit and the central opening required additional gap closing.
- **nsp3:** the central circular wall is covered by the blue exclusion, while outward branches beyond the buffer remain eligible for counting. Isolated bright debris is rejected. The run flags `outer_rim_width_at_limit`. Thin green arcs near the top/right outer edge and some material near the bottom rim may still include rim contamination and need expert review.

These observations come from the saved overlays and are qualitative. No expert-labeled real-image accuracy score is available.

Compared with the archived earlier control result, 150,704 previously rejected candidate pixels are now accepted within the region shared by the old and new analyses. Meanwhile, 288,506 formerly accepted pixels fall inside the new exclusion, and 3,402 formerly accepted pixels are rejected within the shared region. This explains why preserving more branches does not necessarily increase total area: the revised measurement also excludes a larger central band. This comparison combines implementation and configuration changes and is not a ground-truth accuracy assessment.

### Threshold sensitivity

Both foreground and ridge thresholds were varied jointly by ±10%, keeping geometry and other settings fixed. The maximum departure is relative to the baseline area. These values describe parameter sensitivity, not confidence intervals, and do not assess sensitivity to the buffer width or geometry estimation.

| Input | Area at 90% thresholds | Baseline area | Area at 110% thresholds | Maximum departure |
|---|---:|---:|---:|---:|
| Control | 2,622,413 | 2,565,790 | 2,521,330 | 2.21% |
| RGB control | 736,539 | 716,212 | 700,755 | 2.84% |
| nsp3 | 2,612,913 | 2,551,279 | 2,494,727 | 2.42% |

### Observed performance

Images ran sequentially in fresh workers in this Linux workspace. Runtime includes each image's processing, sensitivity runs, and worker audit compression; it excludes final batch archive merging. Peak memory is the worker's peak resident set size. These are observed measurements, not hardware-independent performance guarantees.

| Input | Dimensions | Runtime | Peak memory |
|---|---|---:|---:|
| Control | 5,000 × 5,000 gray | 100.0 s | 690.7 MiB |
| RGB control | 2,488 × 2,520 RGB | 19.4 s | 389.3 MiB |
| nsp3 | 7,917 × 7,917 gray | 448.8 s | 1,162.6 MiB |

The batch was interrupted during nsp3. Its analysis was rerun to completion in a fresh worker, then merged with the two completed results. Configuration and analysis-source checksums match across all three records.

## Output integrity and provenance

The results directory contains exactly five files: the CSV, three labeled overlays, and one audit ZIP. Final verification checked the ZIP, native mask dimensions and binary values, tunnel-mask containment within the analysis region, exact mask counts and reported percentages, mask checksums, source-image checksums, and the shared configuration and code checksum. Source TIFFs remain unchanged.

The analysis-source SHA-256 recorded for all three images is:

```text
c90bd07690a4f2d1683fa04510bfcd7a80bd4b96362cc6a5d39a759201a06327
```

The environment used Python 3.14.2, NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, tifffile 2026.9.20, Pillow 12.3.0, and pytest 9.1.1. See [requirements-lock.txt](requirements-lock.txt) and the audit's `run.json` for reproducibility details.

The earlier 652 generated development files were consolidated into [previous-development-results.zip](archive/previous-development-results.zip), outside the main results folder. Archive integrity was checked before removing those generated working copies. Source images were preserved.

## Independent validation still needed

Freeze the configuration before evaluating held-out expert annotations. Use `--validation-dir` to export six fixed source crops per image; experts should annotate the source crops without viewing predictions. The `validate-crops` command reports agreement with submitted masks. Selected diagnostic crops emphasize difficult cases, so use additional independent sampling or whole-image masks to estimate overall accuracy and systematic area bias.

Remaining limitations include attached debris versus broad junction ambiguity, faint branches, uncertain boundaries, and possible residual outer-rim material. The fixed inner buffer deliberately removes real branch portions as well as circumferential structures. [METHODOLOGY.md](METHODOLOGY.md) defines that measurement and provides text suitable for adaptation into a biology paper.
