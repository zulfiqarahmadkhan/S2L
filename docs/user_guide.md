# S2L user guide

S2L finds cells in microscopy images, measures them, lets you correct the result by hand, trains your own models, and plots the measurements. The sidebar follows that order:

**Home → Segmentation → Correction → Training → Viewer → Plotting → Tools**

At the bottom of the sidebar:

- **Status:** whether a GPU, Cellpose and StarDist were found.
- **GPU | CPU:** GPU is the default. Choose CPU when the GPU is busy with another program. It applies from the next run on and is remembered.
- **Day / Night mode:** switches the colour theme.

---

## Home

Shortcuts to each task, a system status card, and your recent folders.

## Segmentation

1. **Input**
   - **Folder:** every image in a folder.
   - **Spreadsheet (xlsx):** an experiment sheet. Press **Parse**, then tick the wells, stages, channels and types you want.
2. **Output folder:** where masks (`<image>_cp_masks.tif`), Excel sheets and overlay images are written. The input folder is never changed.
3. **Pipeline**
   - **Run segmentation:** find the cells.
   - **Run label → ROI analysis:** measure every cell. This writes one Excel sheet per image and `SummarySheet/Summary.xlsx`.
   - **Segment on / Measure intensity in:** e.g. segment on the phase image (Ph) and measure fluorescence in the red image (R). The matching image is found by well, stage and type in the file name. The hint below shows how many pairs were found.
   - **Skip images that already have a mask:** resume an interrupted run.
   - **Remove cells smaller than … / touching the border:** optional clean-up before measuring.
4. **Model:** a dropdown of the pretrained models, then **Custom (my trained model)**. When Custom is chosen, a second row appears with your trained models (and **Browse** for one trained elsewhere).
   - **Cellpose:** `cpsam` (Cellpose-SAM). Cellpose 4 has only this one pretrained model; it replaces the older cyto / nuclei models. Leave the diameter at "Auto" to detect it automatically.
   - **StarDist:** `2D_versatile_fluo` (fluorescent nuclei), `2D_versatile_he` (H&E, colour images), `2D_paper_dsb2018`, `2D_demo`. StarDist finds nuclei (round objects); for whole cells use Cellpose.
   - **Advanced settings:** thresholds. **Reset to defaults** restores them.
5. **👁 Preview one image:** try the settings on one image and see the cells and the time per image.
6. **Start processing.** **Stop** finishes the current image first.

Each run also saves `run_settings_<date>.json` in the output folder. It records the model, thresholds, channels and whether the GPU or CPU was used.

When the run is done, **🖊 Correct masks** opens the correction page, and **📂 Open output folder** shows the results.

## Correction

Pick an image (or **Choose image…**) and press **Open editor**. The mask is found in the output folder automatically, or you can choose the mask yourself.

Press **F1** in the editor for every shortcut. The main tools:

| Key | Tool | What it does |
|---|---|---|
| P | Pencil | Paint a new cell, or add to the cell you start in. Hold **Ctrl** to paint over neighbours. |
| E | Eraser | Erase part of a cell |
| L | Outline | Draw around a cell; it is filled when you let go |
| W | Magic wand | Click a cell to fill it by brightness |
| A | AI add cell | Draw a box around (or click on) a missed cell, and Cellpose-SAM outlines it |
| D | Delete | Click a cell, or drag a rectangle to delete several |
| F | Fill holes | Click a cell (or use *Fill all holes*) |

Other controls:

- **Undo / redo:** Ctrl+Z and Ctrl+Y.
- **Zoom and pan:** + and − to zoom, Space + drag to pan.
- **Cell information:** hover a cell to see its number, area and intensity. **Show cell numbers** labels all cells.
- **View:** RGB, R, G or B, with intensity, brightness and contrast sliders.
- **Overlay another channel:** show a second channel image on top.
- **Clean-up:** remove small cells or cells touching the border.
- **Saving:** **Ctrl+S** saves in place. **◀ Previous / Next** moves through the folder and saves automatically. After saving, the image's Excel sheet is measured again.

## Training

1. Choose **Cellpose** (uses the GPU) or **StarDist** (CPU on Windows).
2. **Data folder:** images with their masks (`<image>_cp_masks.tif`). Masks corrected in the editor work well as training data.
3. **Training % / Testing %** (80 / 20 by default; they always add up to 100). The testing images are picked at random, never used for training, and saved with the model, so testing later uses exactly the same images. The hint shows how many images go to each.
4. **Start from:**
   - **Cellpose:** `cpsam`, or **Custom (my trained model)** to keep improving one of yours.
   - **StarDist:** **From scratch** (a new network), one of the four pretrained models (fine-tuning), or **Custom (my trained model)**. When fine-tuning, the rays, grid and channels come from the starting model.
5. **Start training.** The loss chart should go down.
6. When training is done, **🧪 Test model** appears. It runs the model on the testing images and saves the results in a new folder: next to the model, or in **Test results folder** if you set one. Tick **Also test the starting model** to get a before / after comparison on the same images. The results include:
   - **`summary_metrics.csv`:** every metric with its 95 % confidence interval:
     - AP, precision, recall and F1 at IoU 0.5 and 0.75;
     - mAP 0.5–0.95;
     - panoptic quality;
     - pixel Dice, Jaccard, sensitivity and specificity;
     - pixel **AUROC** and AUPRC;
     - cell-count error and correlation.
   - **`per_image_metrics.csv`, `metrics_vs_iou_threshold.csv`, `roc_curve.csv`, `pr_curve.csv`, `matched_objects.csv`.**
   - **`figures/`:** PNG (300 dpi) and SVG figures:
     - ROC curve and precision-recall curve;
     - metrics vs IoU threshold;
     - count agreement and IoU histogram;
     - F1 per image;
     - model comparison;
     - training loss.
   - **`examples/`:** each test image with true outlines (green), predicted outlines (magenta) and missed cells (red).
   - **`README.txt`:** what each file means, plus a ready-to-adapt methods paragraph for a paper.
7. **Your trained models** lists everything you trained, with the train / test split and the test AP. **Use in Segmentation** selects a model there; **🧪 Test selected model** tests any listed model again.

## Viewer

Browse a folder of images and masks as thumbnails (tick **Include subfolders** to include them). Click a thumbnail to preview it, scroll to zoom and drag to pan. Use **← Previous / Next →** or the arrow keys to move between images.

## Plotting

- **Distributions (one sheet):** histograms and scatter plots of one Excel sheet. A summary sheet gives one bar per image.
- **Compare groups:** load a results folder, then choose a measurement and group by well, stage, channel, type, image or **condition**. Name conditions like `A6,A7=WT; A8-A10=mutant`.
  - **Plots:** box or violin.
  - **Tests:** Mann-Whitney, Welch t-test or Kruskal-Wallis, with corrected pairwise tests.
  - **Statistical unit:**
    - *Cells* treats every cell as a sample;
    - *Image means* uses one value per image, which is more honest when there are several images per group.
  - **Export table (CSV).**
- **Explore cells:** any measurement against any other, coloured by group, with optional log scales. Hover a point to see the cell; click it to open that cell in the correction editor.

**Save plot…** always saves on a white background (PNG, SVG or PDF), whatever the app theme.

## Tools

- **Image Enhancer:** background subtraction, denoising, contrast (CLAHE), sharpening and 2× upsampling, for one image or a folder.
- **Channel check:** a table of which channel images exist or are missing for every well and stage.
- **Pretrained models:** every Cellpose and StarDist pretrained model, what it is for, and whether it is downloaded. **Download missing models** fetches the rest. The installer already downloads them all, and a missing model is also downloaded the first time it is used.
- **Cellpose GUI / Cell-ACDC:** open these programs. Cell-ACDC is installed automatically the first time.
- **Verify installation:** lists packages and checks the GPU.

---

## Installing and updating

Nothing has to be installed first. If there is no conda, the installer installs Miniforge for you. The first run downloads several GB and takes 10–30 minutes; later starts take seconds.

- **Windows:** double-click `S2L_Setup.exe`. It installs S2L to `%USERPROFILE%\S2L`, adds an **S2L** desktop shortcut, sets everything up and starts the app. Use the shortcut afterwards. If Windows says "Windows protected your PC", click **More info → Run anyway** (the installer is not code-signed).
- **macOS:** unzip `S2L_mac.zip`, drag **S2L.app** to Applications and open it. The first time, macOS blocks it because it is not signed by Apple:
  - **macOS 15 and newer:** go to System Settings → Privacy & Security → **Open Anyway**.
  - **Older macOS:** right-click the app → **Open**.

  A Terminal window then does the setup; keep it open while S2L runs.
- **Developers:** after changing the code, run `python build_installer.py` on Windows. It builds both `S2L_Setup.exe` and `S2L_mac.zip`.
- **GitHub:** `install_and_run.bat` / `install_and_run.sh` install from GitHub (needs Git).
- **PyTorch:** `install_torch.py` picks the build for your NVIDIA driver; see the README.

## Troubleshooting

- **The setup stops with "Required packages could not be installed":** usually a broken download. Run the installer (or the desktop shortcut) again; it continues where it stopped.

- **"Processing failed" with a strange pandas or Excel error after installing something:** restart S2L. Packages changed while it was running.
- **Out of GPU memory:** close other GPU programs, switch to **CPU** in the sidebar, or process fewer or smaller images.
- **Everything is slow:** check the sidebar says "GPU: on". If it says "CPU only", run `python install_torch.py --check`.
- **A crash:** details are written to `s2l_crash.log` in the S2L folder.
