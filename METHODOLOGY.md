# Methodology: visible tunnel coverage, arena-coverage-v1

## Measurement definition

The measured quantity is the number of native-resolution pixels occupied by accepted visible bright tunnel material within a defined analysis region. Branches, loops, and junctions count throughout the arena, including the central interior. Dark gaps between tunnel walls, rejected debris, supported inner-rim material, and outer rim bands do not count. All observed pixels within the fitted arena, including dark central pixels and artifact locations, remain in the coverage denominator. Rejected artifacts and dark pixels contribute no tunnel area. This quantity does not measure tunnel length, lumen area, excavated volume, or physical area without independent calibration.

The processing is deterministic. The shared settings in `config.json` apply by default; reviewed images can explicitly use the restored opening-and-buffer exclusion described below. Independent biological reference masks have not yet been supplied; results remain provisional. [VALIDATION.md](VALIDATION.md) records synthetic accuracy, real-image checks, sensitivity, and performance.

## Inputs, intensity, and scale

Read each TIFF separately with tifffile, using a read-only memory map where possible. Divide unsigned grayscale values by the dtype maximum; invert MINISWHITE data. For RGB, use `(0.2126R + 0.7152G + 0.0722B) / dtype_max`, applied to stored values without an additional gamma transformation.

Geometry and background are estimated on a preview of at most 1,400 pixels per dimension. Large inputs first use a stride to bound the preview construction array, followed by BOX resampling. Native-resolution pixels determine the final mask and area. Preview estimation can miss details below its sampling scale.

Let `D` denote the equivalent diameter of the fitted outer arena in native pixels, and `s = D/1000`. Configured lengths use `s`; configured component areas use `s²`. `inner_rim_fit_tolerance` and `inner_rim_max_width` are fractions of `D`; `inner_rim_intensity` and `inner_rim_min_support` are unitless. TIFF resolution tags are recorded but are not used as specimen calibration.

## Outer arena and central exclusion

1. Identify preview support above `max(0.01, 0.025 × maximum intensity)`. Fit its convex-hull boundary with circle RANSAC (seed 0, 150 trials, tolerance 1.2% of preview extent), then refine an ellipse by bounded geometric least squares with soft-L1 loss. Reject implausible geometry; flag contour support below 80%.
2. Refine the outer scale from intensity drops in 360 native-resolution profiles around the fitted edge. Use the median supported displacement when at least 30 profiles provide a clear drop.
3. Exclude an outer band at least `8s` pixels wide. Trace contiguous bright rim material using 720 angular profiles in a `24s` search band. Bright segments start above 0.20 within `16s` of the fitted edge and end below 0.12. Apply a `1s` guard and periodic median/Gaussian smoothing to reduce the influence of individual radial branches. Flag extensive contact with the search limit. Excluded bands include any tunnel material occupying the same space.
4. Look for an optional central rim. On the preview, threshold at 0.12 and close the bright mask with disk radii corresponding to 2, 4, 6, then 8 pixels per 1,000 diameter pixels. A dark component near the arena centre proposes a rim only if it occupies 1.5–35% of the arena and remains inside normalized arena radius 0.65. Failure to find this proposal no longer fails the measurement.
5. Fit the proposed contour with bounded ellipse least squares. Require minor/major radius ratio at least 0.65, radii between 0.025D and 0.33D, and 90th-percentile fit residual at most `max(1 preview pixel, 0.003D)`. These checks identify only smooth candidates, not biological classes.
6. Examine 720 angular profiles around the candidate. Sample at 0.5 preview-pixel increments, from `-W` to `2W` relative to the candidate ellipse, where `W = 0.012D + 2 preview pixels`. Find contiguous foreground runs above 0.12 starting within `max(1.5 preview pixels, 0.004D)` of the boundary. Reject runs touching the search limits or wider than W. At least 85% of angles must have a plausible run and start/end positions within `max(1.5 preview pixels, 0.003D)` of their respective medians.
7. Use the median start and end plus a guard of `max(1 preview pixel, 0.001D)` to define a narrow rim band. Preserve angles where foreground extends radially on either side of the rim: require at least 65% foreground over a neighbouring interval twice the typical rim width. Pad protected angles by the guard converted to angle using the smaller ellipse radius. Missing and protected angles are not excluded. This can retain some rim pixels at genuine branch junctions.
8. Evaluate that band at native pixel centres using the preview-to-native coordinate transform, independent of tile boundaries. Exclude only supported angles of this band. Never fill the central interior or apply the former `0.025D` buffer. Morphological closing is only for proposing geometry and never adds measured pixels.

The segmentation-eligibility region `R` is the arena after its outer-rim exclusion, minus the supported narrow inner-rim band if detected. This determines where tunnel pixels can be accepted, but is not the coverage denominator. The central interior is always eligible. A detected rim receives `inner_rim_exclusion_requires_review`; otherwise the analysis continues with `inner_rim_not_identified_review_centre`. Non-detection is not confirmation of absence. A smooth central tunnel loop can resemble a rim, and partial, irregular, or branch-interrupted rims can remain unremoved. These limitations require visual review and expert validation.

This rule replaces the earlier buffered central-opening definition. The central-tunnels-v1 revision changed both numerator and denominator. The current arena-coverage-v1 revision preserves that segmentation and changes normalization to the full fitted arena; regenerate all samples for a comparison under one frozen revision. Historical buffered-centre outputs remain valid records of their original definition and are not interchangeable with revised measurements.

## Background and foreground candidates

Estimate preview background `B` using a square grayscale opening with half-width `12s`, followed by Gaussian smoothing with sigma `4s` (converted to preview coordinates). Interpolate `B` bilinearly into native tiles. Compute corrected intensity `J = max(I - B, 0)`.

Calculate Otsu's threshold on preview `J` inside `R`, and estimate noise as `1.4826 × MAD` of the lower half of those values:

```text
T = max(0.06, 0.65 × Otsu(J within R), 4 × noise)
C = R AND (J >= T)
```

Clipped black images can have a zero noise estimate; the floor and Otsu term remain active. All recovered tunnel pixels must belong to candidate mask `C`.

## Multiscale ridge support

Compute Gaussian Hessians at scales `max(0.7, 0.5s)`, `max(0.7, 1s)`, `max(0.7, 2s)`, and `max(0.7, 4s)`. Normalize derivatives by sigma squared. For eigenvalues ordered by magnitude, use a bright-ridge Frangi response:

```text
|lambda1| <= |lambda2|
Rb = |lambda1| / (|lambda2| + 1e-12)
S2 = lambda1² + lambda2²
V_sigma = exp(-Rb²/(2 × 0.5²)) × [1 - exp(-S2/(2 × 0.06²))]
V_sigma = 0 if lambda2 >= 0
V = maximum response across scales
```

This implementation uses SciPy Gaussian derivatives with finite support (`truncate=4`) and explicit scale normalization. It differs from simply calling scikit-image's Frangi function with default normalization and automatic gamma. The multiscale formulation follows [Frangi et al., 1998](https://doi.org/10.1007/BFb0056195); [scikit-image's ridge documentation](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi) describes the related filters. The derivative operations are documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

Candidate pixels with `V >= 0.12` are seeds. Initially retain candidates within `max(1, 3s)` of a seed. Distance-bounded support avoids accepting an entire connected component solely because it touches a ridge.

## Branch continuity and debris rejection

Width is an inspection trigger, not a sufficient reason to delete a branch. The shape mask combines candidates with original intensities at least 0.65 inside `R`; this prevents background subtraction from disguising a solid white blob as a hollow ring.

Calculate local half-width as `max(distance_to_background_pixel_center - 0.5, 0)`. The half-pixel correction expresses width relative to foreground boundaries. Inspect localized cores wider than `max(2, 8s)`. Extended cores longer than four inspection radii retain ordinary ridge support rather than being erased. Bounded elongated patches with moment-axis ratio at least 3 are preserved.

For a compact suspicious patch:

- Skeletonize its local shape mask and identify incoming arms that touch the patch and extend at least 1.5 inspection radii into surrounding foreground.
- With two or more supported arms, retain the skeleton paths connecting them. Iteratively prune dead-end skeleton branches without an incoming-arm anchor.
- Estimate incoming half-widths from skeleton samples outside the patch. Use their 75th percentile, with an allowance of 1.15 for two arms or 1.45 for junctions, to form a connecting corridor. Add the half-pixel boundary offset when rasterizing that corridor.
- Preserve foreground candidates within the corridor. Reject unsupported expansion outside it. A one-arm compact blob is not automatically classified as a through-branch.

This restores a branch through a suspicious patch while limiting attached mass. It also avoids filling dark gaps: recovered area remains a subset of `C`.

Apply global 8-connected component filtering after continuity repair. Remove components smaller than `max(2, 6s²)`. Retain components with moment-axis ratio at least 2.5, or extended sparse networks with major-axis extent at least `35s` and moment-ellipse occupancy below 0.45. Add 0.25 pixel² to moment variances to stabilize ratios for thin objects. Disconnected elongated fragments can count; connection to the opening or outer edge is not required.

## Measurement, outputs, and sensitivity

```text
Tunnel_area_px = sum(final_mask)
A = observed native pixel centres inside the fitted outer ellipse (normalized radius < 1)
Arena_area_px = sum(A)
Analysis_region_area_px = Arena_area_px  # compatibility alias
Tunnel_area_percent = 100 × Tunnel_area_px / Arena_area_px
```

The denominator A includes the centre, dark spaces, and locations of rejected inner/outer rim material or debris. It is independent of the angular rim cutoffs and segmentation decisions. It is rasterized in bounded native-resolution strips; no pixels beyond the image are extrapolated. Estimate missing fraction as `max(0, 1 − Arena_area_px / (πab))`, where a and b are native ellipse radii. The analytic ellipse area is only a cropping diagnostic, never the denominator. If this fraction exceeds `arena_missing_fraction_warning = 0.005`, flag `arena_boundary_clipped_review_coverage`; retain the observed-region value in the CSV for diagnosis but omit it from the comparison graph. Boundary identification is an image-based approximation of the experimental surface, not a physical calibration.

Each intersection pixel counts once. No skeleton length or filled tunnel footprint is substituted for area. Independent physical x/y pixel sizes could later convert pixel area by their product; the current output is in pixels.

Default image/folder results consist of one CSV, one labeled overlay per input, and one audit ZIP. Resumable batch results also include a single-panel coverage PNG generated from the CSV. Green overlay pixels show accepted foreground, orange shows rejected candidates, and blue shows exclusions. Reduced previews use maximum occupancy to keep thin structures visible; exact native 0/1 tunnel, full-arena denominator (`analysis_region`), and segmentation-eligibility (`segmentation_region`) masks are archived. The white overlay boundary shows the denominator arena; blue rim exclusions affect the numerator, not the denominator. Intermediate files are temporary. Diagnostic images and annotation crops require explicit output directories outside the results.

The audit includes image and code checksums, mask checksums, full/effective parameters, detected geometry, versions, UTC start time, worker runtime, peak RSS, flags, and sensitivity. Folder processing uses fresh sequential workers. Per-image runtime includes analysis, sensitivity and worker audit compression; it excludes the batch archive merge.

Sensitivity repeats classification with both intensity and ridge thresholds multiplied jointly by 0.9 and 1.1, holding geometry and filtering fixed. Report each area and maximum relative departure from baseline; flag departures above 15%. This is parameter sensitivity, not statistical confidence. Inner-rim geometry and width parameters are not varied in this check; this sensitivity result does not quantify boundary uncertainty.

## Validation and limits

Synthetic masks provide independently known tunnel foreground; central tunnel pixels remain in the reference. Tests cover clean images, noise and shading, broad tunnels, loops, double walls, branches, junctions, isolated and attached debris, and circular rims. Required performance is precision and recall of at least 95%, with absolute relative area error no greater than 5%. Ring-only images must give zero area. Implementation tests cover missing and fragmented rims, central tunnel preservation, rim/branch connections, native tile consistency, continuity, exact counting, audit/overlay agreement, input preservation, deterministic behavior, tile boundaries, failed inputs, and the compact output contract.

Synthetic accuracy is not biological accuracy. Width and topology cannot always distinguish broad biological junctions from attached debris. A curved artifact can resemble a branch. Abrupt true branch endings, faint material, large solid patches with line-like edges, and uncertain arena boundaries remain possible errors. The initial stride used for preview construction and the estimated opening contour can affect exclusions.

Earlier settings were developed using synthetic tests and visual inspection of three images. The revised central rule passes synthetic checks, but its conservative detector did not confidently identify an inner rim in any of those three local previews. No revised native biological measurements have been validated. Freeze and archive settings before an independent, held-out expert evaluation. Optional diagnostic crops emphasize difficult cases and cannot estimate unbiased whole-image accuracy. Experts should annotate source crops without predictions, record uncertain areas, and assess agreement and systematic area bias. Cross-image biological comparisons should use independently verified calibration or the consistently defined coverage measure, with appropriate biological replication.

## Draft text for a biology paper

> Visible tunnel-network area was quantified from native-resolution TIFF images using a Python workflow. Genuine tunnels were eligible throughout the arena, including its central region. A conservative geometric detector proposed smooth inner-rim exclusions using angular foreground support and branch-continuity checks; absent or ambiguous central boundaries did not prevent measurement and were flagged for review. Outer rim bands were excluded. Following background correction, intensity thresholding and multiscale Hessian ridge evidence identified candidate tunnel material. Local branch-continuity and width criteria retained connecting branches and junctions while reducing compact debris. Area was calculated from accepted native pixels and expressed as a percentage of the observed area inside the fitted arena. The normalization region included the central interior and was independent of artifact rejection. Overlays, masks, parameters, and provenance were archived. Classification sensitivity was assessed by jointly varying intensity and ridge thresholds by ±10%. Independent expert validation is required to establish biological segmentation accuracy.

Add actual expert-validation results and physical calibration only after those steps are completed.

## Software references

- [Frangi et al. (1998), Multiscale vessel enhancement filtering](https://doi.org/10.1007/BFb0056195).
- [scikit-image filters, including Otsu](https://scikit-image.org/docs/stable/api/skimage.filters.html), and [model estimation/RANSAC](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac).
- [SciPy Gaussian derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) and [Euclidean distance transforms](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html).
- [tifffile scientific TIFF support](https://github.com/cgohlke/tifffile).
- [NumPy citation guidance](https://numpy.org/citing-numpy/).

## Figure export

The batch comparison shows a horizontal bar per image with a zero-based linear percentage axis. Matplotlib renders a 7.1-inch-wide, 300-dpi PNG and an editable SVG archived under `figures/`. Font size and figure height are chosen for readable labels. Failed/invalid values and flagged cropped arenas have labeled rows without bars. Exact plotting values, axis limits, original filenames and simplified display labels, and a draft caption are archived. PNG and SVG checksums are verified at completion/resume. No pixel-area panel, replicate confidence intervals, or significance tests are displayed. The CSV records the rule identifier `arena-coverage-v1` to prevent old percentages being mislabeled as full-arena coverage.

## Explicit restoration of an inner opening

For an image confirmed to contain a central opening, `inner_exclusion="opening-buffer"` restores the earlier rule. Threshold the preview at 0.025 and close with disks of radius 2, 4, 6, then 8 pixels per 1,000 arena diameter pixels. Select the most frequent nonzero dark-component label in the 7×7 window about the fitted centre. Accept the first component occupying 1.5–35% of the outer-band-trimmed arena and confined to normalized arena radius ≤0.65. Fill holes in this opening mask. If no component qualifies, fail the image rather than assume the rim is absent.

At native resolution, nearest-neighbour sampling transfers the opening mask. Refine its one-preview-pixel uncertainty band using native intensities below 0.025, retaining the eroded core. Exclude the refined opening and pixels with Euclidean distance ≤0.025D from it. This geometric exclusion removes any tunnel portions within the buffer as well as rim artifacts; it does not claim to distinguish their biology. The full fitted arena denominator remains unchanged. The existing default `rim-only` treatment continues to allow genuine central tunnels where the opening has washed away.

Selected-image revisions inherit all other per-image settings and sensitivity checks. Unselected records and mask/overlay contents are retained, and the combined descriptive graph is regenerated. Parent-run checksums and per-image settings and software identities document this deliberate difference in exclusion policy. Cross-image comparisons require review of these policies; independently annotated reference masks are still needed for biological validation.
