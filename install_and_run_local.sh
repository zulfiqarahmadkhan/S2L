#!/usr/bin/env bash
# S2L setup + launcher for macOS and Linux. Works on a new computer: installs
# Miniforge (conda) if no conda is found, creates the S2L environment, installs
# PyTorch, all packages and the pretrained models, then starts S2L.
# Later runs only check that everything is in place (a few seconds).
#
# Optional settings (environment variables, mainly for testing):
#   S2L_ENV_NAME    conda environment name           (default S2L)
#   S2L_CONDA_HOME  use / install conda in this folder only
#   S2L_NO_LAUNCH   set to 1 to set up without starting the app
set -u
cd "$(dirname "$0")"
ENV_NAME="${S2L_ENV_NAME:-S2L}"
PYTHON_VER="3.11"
NEW_CONDA="${S2L_CONDA_HOME:-$HOME/miniforge3}"

fail() { echo ""; echo "[ERROR] $*"; echo ""; read -r -p "Press Enter to close…" _ || true; exit 1; }

echo ""
echo "  ========================================"
echo "          S2L setup and launcher"
echo "  ========================================"
echo ""

# ─── Find conda (Anaconda, Miniconda or Miniforge) ──────────────────────
CONDA=""
if [ -n "${S2L_CONDA_HOME:-}" ]; then
    [ -x "$S2L_CONDA_HOME/bin/conda" ] && CONDA="$S2L_CONDA_HOME/bin/conda"
else
    for c in "$(command -v conda 2>/dev/null || true)" \
             "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" \
             "$HOME/anaconda3/bin/conda" "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
             "/opt/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" "/usr/local/anaconda3/bin/conda"; do
        if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
    done
fi

# ─── No conda: install Miniforge for this user (no admin rights needed) ──
if [ -z "$CONDA" ]; then
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64)  MF="Miniforge3-MacOSX-arm64.sh" ;;
        Darwin-x86_64) MF="Miniforge3-MacOSX-x86_64.sh" ;;
        Linux-x86_64)  MF="Miniforge3-Linux-x86_64.sh" ;;
        Linux-aarch64) MF="Miniforge3-Linux-aarch64.sh" ;;
        *) fail "Unsupported system $(uname -s) $(uname -m)." ;;
    esac
    echo "[*] No conda found. Installing Miniforge to $NEW_CONDA (about 100 MB, one time)..."
    TMP_SH="$(mktemp -t miniforge.XXXXXX).sh"
    curl -L --fail --silent --show-error -o "$TMP_SH" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/$MF" \
        || fail "Could not download Miniforge. Check the internet connection."
    bash "$TMP_SH" -b -p "$NEW_CONDA" || fail "Miniforge installation failed."
    rm -f "$TMP_SH"
    CONDA="$NEW_CONDA/bin/conda"
    echo "[*] Miniforge installed."
fi
echo "[*] Using conda: $CONDA"

# ─── S2L environment ─────────────────────────────────────────────────────
if ! "$CONDA" env list | grep -q "^${ENV_NAME} "; then
    echo "[*] Creating the '$ENV_NAME' environment with Python $PYTHON_VER (one time)..."
    # conda-forge only: works with every conda and needs no Anaconda terms-of-service acceptance
    "$CONDA" create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VER" \
        || fail "Could not create the conda environment '$ENV_NAME'."
    rm -f .s2l_req_done
fi
run() { "$CONDA" run --no-capture-output -n "$ENV_NAME" "$@"; }

# ─── PyTorch (CUDA build on Linux with an NVIDIA GPU; Apple GPU via MPS on Macs) ─
echo "[*] Checking PyTorch..."
run python install_torch.py || echo "[WARN] PyTorch setup reported a problem; S2L may run on the CPU."

# ─── Packages (skipped when unchanged since the last successful install) ─
cat requirements.txt requirements-optional.txt > .s2l_req_now
if ! cmp -s .s2l_req_now .s2l_req_done 2>/dev/null; then
    echo "[*] Installing required packages (first run: several minutes)..."
    run python -m pip install --quiet --disable-pip-version-check -r requirements.txt \
        || { echo "[*] Retrying once (downloads can fail on a slow or busy connection)..."; \
             run python -m pip install --quiet --disable-pip-version-check -r requirements.txt; } \
        || fail "Required packages could not be installed. Check the messages above."
    echo "[*] Installing optional packages (StarDist, TensorFlow, Cell-ACDC)..."
    run python -m pip install --quiet --disable-pip-version-check -r requirements-optional.txt \
        || echo "[WARN] Some optional packages failed; S2L works without them (StarDist may be unavailable)."
fi
if ! run python -c "import PyQt6.QtWidgets, cellpose.models, torch, sklearn, skimage, pandas"; then
    rm -f .s2l_req_done
    fail "S2L's packages do not import. Run this again; if it keeps failing, check the messages above."
fi
mv -f .s2l_req_now .s2l_req_done

# ─── Pretrained models (Cellpose-SAM + StarDist; skipped when present) ───
echo "[*] Checking pretrained models (the first run downloads about 1.2 GB)..."
run python -m s2l.download_models || echo "[WARN] Some models could not be downloaded; they are downloaded on first use."

if [ "${S2L_NO_LAUNCH:-}" = "1" ]; then
    echo "[*] Setup complete (not launching: S2L_NO_LAUNCH=1)."
    exit 0
fi

# ─── Launch ──────────────────────────────────────────────────────────────
echo "[*] Starting S2L... (keep this window open while S2L runs)"
echo ""
run python main.py || fail "S2L exited with an error. Details are in s2l_crash.log in $(pwd)."
