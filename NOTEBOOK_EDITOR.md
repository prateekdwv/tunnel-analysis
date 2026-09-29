# Review one image in Colab

This is an experimental workflow on `feature/notebook-mask-editor`. Local tests cover mask arithmetic and widget event handlers. **Live Colab rendering and responsiveness still need a browser check.** The existing batch notebook and commands remain available.

## Start

Open [Google_Drive_Mask_Editor.ipynb](notebooks/Google_Drive_Mask_Editor.ipynb). After the branch is pushed, use [Open in Colab](https://colab.research.google.com/github/prateekdwv/tunnel-analysis/blob/feature/notebook-mask-editor/notebooks/Google_Drive_Mask_Editor.ipynb).

1. Use a CPU runtime. Set the input folder and results parent. Leave `RESUME_DRAFT` blank for a new image. Keep `RUN_SENSITIVITY=False` for the quicker interactive workflow.
2. Connect Drive and install the selected Git revision. The notebook installs into its own kernel so widget callbacks can access the masks. Changing package versions requires a runtime restart. Resume installs the saved commit and recorded package versions; a different Python version is rejected.
3. Choose one TIFF from the dropdown and open it. Save the printed draft path.
4. Try the brush demo **before analysis**: paint, erase, radius 1, undo, and check pointer alignment. It uses a real crop but never saves measurements. Stop if the canvas is blank or inaccurate.
5. Adjust and confirm the boundaries. Then run automatic analysis and open the correction editor.

Run cells individually, because boundary confirmation and review require your input. This notebook does not monitor Drive or launch a remote desktop.

## Accurate boundaries

There are three independent ellipses:

- **Full arena:** defines the percentage denominator. Fit the visible arena itself.
- **Outer cutoff:** excludes the outer rim from counted tunnels.
- **Inner exclusion:** excludes its entire interior; enable it only when appropriate.

Move or resize the selector, use the angle field for rotation, or constrain it to a circle. Coordinates and radii are native image pixels. Rotation is displayed in degrees, clockwise in image coordinates. Equivalent ellipses may swap radii to keep the selector's angle within its supported ±45° range.

The initial outer fit is only a suggestion. The inner exclusion starts disabled, with a placeholder ellipse available to draw/adjust. A failed fit still permits manual geometry. Ellipses need not share a centre. Exclusions are intersected with the full arena.

Use the toolbar to pan towards an edge and **Inspect selected edge** to load a native 512-pixel crop around the nearest point on the selected ellipse. Pan to other edges and repeat. **Whole image** restores the overview. Confirm boundaries before analysis. An ellipse cannot exactly follow an irregular rim; assess any residual rim or unwanted tunnel exclusion rather than assuming the shape is accurate.

Changing boundaries invalidates the automatic result. Apply/discard tunnel strokes first. Earlier applied revisions remain in the hidden draft history; a new analysis starts a new automatic mask and does not transfer old manual strokes across changed boundaries.

## Correct tunnels

Select an origin in the overview, choose a crop size (256, 512 or 1024), and click **Open crop**. The selection controls the origin; the crop-size field controls its extent. Crops are clipped at image edges. Integer zoom factors 1, 2 and 4 enlarge the display without resizing the measurement mask. To bound rendering memory, the enlarged canvas is limited to 1024 pixels per side: use a 256-pixel crop for 4× zoom, 512 for 2×, or 1024 for 1×.

- **Paint / Erase:** direct binary edits. Radius 1 at an integer pixel centre changes one pixel. Hold Alt for temporary erasing.
- **Show mask:** toggle green/blue overlays to see underlying tunnel edges. Green counts; blue is excluded. Painting never adds pixels in blue areas.
- **Undo stroke:** undo up to 20 strokes in the current crop. Switching or applying crops clears this undo history.
- **Apply to Drive:** save a compressed crop patch and verify the checkpoint. A failed write leaves strokes available to retry.
- **Discard strokes:** restore the last applied crop.

The editor does not fill enclosed spaces or snap brush strokes to intensity boundaries. Additions may recover pixels missed by thresholding, so carefully trace visible material and preserve dark gaps. The overview is a display preview; inspect native crops for exact edges.

## Save and resume

Only applied edits survive a disconnect. Paste the printed draft directory into `RESUME_DRAFT` and restart from setup; source checksums, configuration, code and dependencies must match. Do not open the same draft for writing in two notebooks. Draft objects are compressed and immutable; only a verified commit makes a new revision visible. Interrupted temporary uploads are ignored; missing or corrupt committed objects cause an explicit error.

Click **Save reviewed result** after reviewing the whole image. Each click creates a new dated folder with exactly:

```text
results.csv
overlay.png
audit.zip
```

The CSV gives exact native pixel area and coverage of the full arena. The overlay remains a preview. The audit stores automatic/corrected masks, candidates, arena and permitted-region masks, crop edits, geometry, reviewer, source checksum, code/dependency identity, runtime and sensitivity settings. The source TIFF is never modified. Old results and applied draft history remain preserved.

`reviewed_provisional` records operator review, not independent biological validation. Sensitivity, when enabled, describes the automatic mask before manual correction.

## Compare selected results

List one reviewed result folder per image in the final notebook cell, or run:

```bash
tunnel-analysis aggregate-reviewed "Reviewed/reviewed_..." "Reviewed/reviewed_..." --output "Reviewed/comparison_new"
```

This verifies native masks and measurements, then writes the combined CSV, existing graph design as PNG/SVG, and source provenance. It refuses duplicate images, overwrites and older automatic measurement definitions. It does not infer which revision you intended to compare.

## Local development

```bash
python -m pip install -e '.[notebook,test]'
python -m pytest
```

Processing/persistence functions are in `tunnel_analysis.editor`; widgets are optional in `tunnel_analysis.notebook_editor`. Use `EditSession`, `confirm_boundaries`, `analyse`, `crop`, `Crop.apply`, `save_reviewed`, and `aggregate_reviewed` outside Colab with ordinary directories. Call `session.close()` to release local scratch storage after saving. Widget dependencies are not required by the existing CLI.

Stackview 0.19.1 is pinned because the adapter uses its annotation controls and stroke interpolation with a local clipped-brush override. It also adds pointer-down/up handling so isolated clicks and undo have defined behavior. No global library monkeypatches are applied. Upgrade only after the adapter's event and zoom tests pass.
