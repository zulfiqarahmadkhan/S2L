# S2L

Cell segmentation and ROI analysis tool. Supports Cellpose and StarDist segmentation engines with a modern PyQt6 interface.

## Features

- **Segmentation** — Run Cellpose (including SAM) or StarDist on microscopy images
- **ROI Analysis** — Convert segmentation masks to quantitative measurements (area, integrated density, mean gray value, standard deviation) exported as Excel spreadsheets
- **Dataset Viewer** — Browse large image folders with lazy-loaded thumbnails, zoom/pan preview, and keyboard navigation
- **Spreadsheet Import** — Parse experiment xlsx files (FIM sheet format) to extract image paths with well/channel/stage filters
- **Multi-channel experiments** — Segment on one channel (e.g. Ph) and measure intensity in the matching image of another (e.g. R), paired automatically by well / stage / type; preview settings on one image; skip already-finished images; optional size and border filters; every run's settings saved as JSON
- **Manual Correction** — Pencil, partial eraser, outline-and-fill, magic wand, **AI add cell** (Cellpose-SAM on a box or click), delete (click or rectangle), fill holes; undo / redo; cell info on hover and cell numbers; RGB / R / G / B views with intensity, brightness and contrast; second-channel overlay; clean-up filters; Previous / Next through a folder with autosave and automatic re-measurement (F1 in the editor lists all shortcuts)
- **Models** — Choose a pretrained model (Cellpose-SAM; StarDist 2D_versatile_fluo, 2D_versatile_he, 2D_paper_dsb2018, 2D_demo) or "Custom (my trained model)"; all pretrained models are downloaded by the installer
- **Training** — Fine-tune Cellpose (GPU) or train StarDist (from scratch, from a pretrained model, or from your own model); choose Training % / Testing %; live loss chart; list of your trained models that can be sent straight to Segmentation
- **Model testing** — "Test model" evaluates on the held-out test images and saves paper-ready results: AP / precision / recall / F1 at IoU 0.5 and 0.75, mAP 0.5:0.95, panoptic quality, pixel Dice / Jaccard / AUROC / AUPRC, count error, all with 95 % bootstrap CIs (CSV), plus ROC, PR and other figures (PNG 300 dpi + SVG) and example overlays
- **Plotting** — Distributions of one sheet; **compare groups** (wells, stages, or your own conditions such as "A6,A7=WT; A8-A10=mutant") with box / violin plots, Mann-Whitney / t-test / Kruskal-Wallis, and CSV export; **explore cells** interactively — hover a point to see the cell, click to open it in the correction editor
- **Channel check** (Tools) — Table of which channel images exist / are missing for every well before a run
- **Image Enhancer** (Tools) — Background subtraction, denoising, contrast (CLAHE), sharpening and 2× upsampling for one image or a whole folder
- **Day / night theme** — Toggle at the bottom of the sidebar; remembered between sessions

## Quick Install

**Which file do I run?**

| Computer | File | What to do |
|---|---|---|
| **Windows** | `S2L_Setup.exe` | Double-click it. Next time, start S2L from the **S2L** shortcut on the desktop. |
| **Mac** | `S2L_mac.zip` | Double-click to unzip, drag **S2L.app** into Applications, then open it. |

The other files (`.bat`, `.sh`, `.py`) are used by the installers behind the scenes; you don't need to run them.

Nothing needs to be installed first: if the computer has no conda (Anaconda, Miniconda or
Miniforge), the installer installs **Miniforge** (free, conda-forge) for the current user, no
administrator rights needed. The first run downloads several GB (Python, PyTorch, the pretrained
models) and takes 10–30 minutes; later starts take seconds.

### Windows — `S2L_Setup.exe`

Double-click **`S2L_Setup.exe`**. It copies S2L to `%USERPROFILE%\S2L`, puts an **S2L** shortcut on
the desktop, then sets everything up and starts the app:

1. conda (Miniforge, if none is installed)
2. the `S2L` environment with Python 3.11
3. the PyTorch build that matches the NVIDIA driver (CPU build if there is no NVIDIA GPU)
4. all packages, then the optional ones (StarDist / TensorFlow, Cell-ACDC) — S2L still starts if an optional one fails
5. the pretrained models (Cellpose-SAM and the StarDist models)

Use the desktop shortcut to start S2L afterwards. Windows may show "Windows protected your PC"
the first time, because the installer is not code-signed: click **More info → Run anyway**.

### macOS — `S2L_mac.zip`

Unzip it, drag **S2L.app** to Applications and open it. A Terminal window opens and does the same
steps as on Windows (PyTorch with Apple-GPU support on Apple-silicon Macs). The app is not signed by
Apple, so macOS blocks it the first time:

- macOS 15 (Sequoia) and newer: open **System Settings → Privacy & Security** and click **Open Anyway**.
- macOS 14 and older: right-click **S2L.app → Open → Open**.

The zip contains "READ ME FIRST (Mac).txt" with these steps. S2L is installed to `~/S2L`.

### From GitHub (Windows, macOS, Linux)

`install_and_run.bat` (Windows) or `install_and_run.sh` (macOS / Linux) clone or update S2L from
GitHub into `~/S2L` and then run the same setup (`install_and_run_local.bat` / `.sh`). Needs Git.

```bash
curl -O https://raw.githubusercontent.com/we-02/S2L/main/install_and_run.sh
bash install_and_run.sh
```

### Building the installers

After changing the code, rebuild both installers on Windows:

```
python build_installer.py
```

This writes `S2L_Setup.exe` (compiled with the C# compiler that ships with Windows) and
`S2L_mac.zip` (no Mac needed to build it). The setup scripts accept `S2L_ENV_NAME`,
`S2L_CONDA_HOME` and `S2L_NO_LAUNCH=1` for testing an install in a separate folder.

## Manual Install

If you prefer to set things up yourself:

```bash
git clone https://github.com/we-02/S2L.git
cd S2L
conda create -n S2L -c conda-forge --override-channels python=3.11 -y
conda activate S2L
python install_torch.py
pip install -r requirements.txt
pip install -r requirements-optional.txt   # StarDist, TensorFlow, Cell-ACDC
python -m s2l.download_models               # pretrained models (optional, else on first use)
```

Install PyTorch **before** `requirements.txt`. Otherwise pip installs the CPU-only PyTorch from PyPI, and a later CUDA install does nothing because pip thinks PyTorch is already installed.

Then run:

```bash
python main.py
```

## GPU / PyTorch

`install_torch.py` reads the CUDA version your NVIDIA driver supports (the `CUDA Version` shown by `nvidia-smi`). It then installs the newest PyTorch build that driver can run:

| Driver supports CUDA | PyTorch build installed |
|---|---|
| 13.0 or newer | `cu130` |
| 12.8 – 12.9 | `cu128` |
| 12.6 – 12.7 | `cu126` |
| 11.8 – 12.5 | `cu118` (older PyTorch versions only) |
| older, no NVIDIA GPU, or macOS | CPU |

You don't need to install the CUDA Toolkit separately; the PyTorch wheels include the CUDA libraries they need. To get a newer CUDA build, update the NVIDIA driver and run the script again.

```bash
python install_torch.py --check   # show the installed and recommended build
python install_torch.py           # install / switch to the recommended build
python install_torch.py --cpu     # force the CPU-only build
```

If you answer "n" to the CUDA question in the launcher, it creates a `.s2l_torch_cpu` file in the install folder and keeps using the CPU build. Delete that file to switch to the GPU build.

StarDist uses TensorFlow, which has no native GPU support on Windows, so StarDist runs on the CPU there. Cellpose uses the GPU.

**GPU / CPU switch.** When a GPU is found, the sidebar shows **GPU | CPU** buttons. GPU is the default; choose CPU when the GPU is busy with another program. The choice is remembered and applies to segmentation, Cellpose training and the AI add-cell tool from the next run on. Each run's `run_settings_<date>.json` records which one was used.

## Usage

A full walk-through of every page is in [docs/user_guide.md](docs/user_guide.md).

### Folder Mode

1. Open the **Segmentation** page
2. Set input mode to **Folder**
3. Browse for your image directory and output directory
4. Choose an engine (Cellpose or StarDist) and model
5. Enable "Run segmentation" and/or "Run label → ROI analysis"
6. Click **Start processing**

### Spreadsheet Mode

1. Set input mode to **Spreadsheet**
2. Browse for your experiment `.xlsx` file
3. Click **Parse spreadsheet**
4. Use the Well / Stage / Channel / Type checkboxes to filter images
5. Set an output directory
6. Click **Start processing**

Masks are saved to the output directory. The ROI step generates Excel files and overlay images there as well.

### Dataset Viewer

Navigate to the **Viewer** page to browse image folders. Click **Open folder** or use the **View** buttons next to directory inputs on other pages. Supports:

- Thumbnail grid with lazy loading
- Click to preview, scroll to zoom, drag to pan
- Arrow keys to navigate, Escape to deselect
- Filename filter bar

## Project Structure

```
S2L/
├── main.py                       # Entry point (python main.py)
├── requirements.txt              # Python packages (install PyTorch first, see above)
├── install_torch.py              # Installs the PyTorch build that matches the NVIDIA driver
├── verify_installation.py        # Package / GPU check (also Tools → Verify installation)
├── requirements-optional.txt     # StarDist / TensorFlow / Cell-ACDC (S2L runs without them)
├── install_and_run_local.bat     # Windows setup + launcher (installs conda if needed)
├── install_and_run_local.sh      # macOS / Linux setup + launcher
├── install_and_run.bat / .sh     # Clone / update from GitHub, then run the setup above
├── build_installer.py            # Builds S2L_Setup.exe and S2L_mac.zip from the current code
├── S2L_Setup.exe                 # Windows installer
├── S2L_mac.zip                   # macOS app (S2L.app)
├── app_icon.ico
├── docs/
│   └── user_guide.md             # How to use every page
└── s2l/
    ├── core/                     # Processing — no GUI code
    │   ├── segmenter.py          # Cellpose-SAM segmentation
    │   ├── stardist_segmenter.py # StarDist segmentation
    │   ├── roi_converter.py      # Masks → per-cell measurements (Excel) + overlay image
    │   ├── summary.py            # Summary sheet of one run
    │   ├── spreadsheet_parser.py # Experiment .xlsx → image list with well / stage / channel tags
    │   ├── channels.py           # Pairs channel images (segment on Ph, measure in R, …)
    │   ├── image_files.py        # Image / mask file naming rules
    │   ├── trainer.py            # Cellpose fine-tuning
    │   ├── train_job.py          # Runs a training in its own process (keeps the app stable)
    │   ├── stardist_trainer.py   # StarDist training (from scratch, pretrained or own model)
    │   ├── datasplit.py          # Training / testing split, saved with each model
    │   ├── evaluation.py         # "Test model": metrics (CSV), figures, overlays (own process)
    │   ├── mask_edit.py          # Editing operations of the correction editor
    │   ├── ai_assist.py          # AI "add this cell" (Cellpose-SAM on a box or click)
    │   ├── enhance.py            # Image Enhancer filters
    │   ├── results.py            # Loads a results folder; group comparisons and statistics
    │   ├── device.py             # GPU / CPU choice
    │   └── models_info.py        # Pretrained model list, downloads, environment probe
    ├── download_models.py        # Downloads all pretrained models (run by the installers)
    ├── ui/
    │   ├── main_window.py        # Window, sidebar, GPU / CPU and day / night switches
    │   ├── pages/                # One file per sidebar page
    │   │   ├── home.py
    │   │   ├── segmentation.py
    │   │   ├── correction.py
    │   │   ├── training.py
    │   │   ├── plotting.py
    │   │   └── tools.py
    │   ├── dataset_viewer.py     # Viewer page
    │   ├── manual_correction.py  # Mask correction editor
    │   ├── workers.py            # Background threads (segmentation, training, environment check)
    │   ├── widgets.py            # Shared widgets, settings, error dialogs
    │   └── theme.py              # Day / night colour palettes and stylesheet
    └── utils/
        └── sam_utils.py          # Cellpose-SAM availability check
```

Settings (last folders, theme, GPU / CPU choice, trained-model list) are stored per user by Qt (`HKEY_CURRENT_USER\Software\S2L` on Windows), not in the project folder.

## Requirements

- Python 3.11
- PyQt6
- NumPy < 2.1 (for numba/stardist compatibility)
- OpenCV, scikit-image, scipy
- pandas, openpyxl, xlsxwriter
- Cellpose >= 4.0
- PyTorch >= 2.0
- StarDist (optional)
