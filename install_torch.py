"""Install the PyTorch build that matches this machine's NVIDIA driver.

Run with the S2L environment's Python (the launchers do this for you):

    python install_torch.py          # pick the CUDA build the driver supports, else CPU
    python install_torch.py --cpu    # force the CPU-only build
    python install_torch.py --check  # only report what is installed and what would be chosen

The newest CUDA build the driver supports is chosen from the PyTorch wheel
indexes. Nothing is reinstalled when the right build is already working.
"""
import argparse
import json
import platform
import re
import subprocess
import sys

INDEX = "https://download.pytorch.org/whl/{}"
# (minimum driver CUDA version, wheel tag), newest first
CUDA_BUILDS = [((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"), ((11, 8), "cu118")]


def driver_cuda_version():
    """Highest CUDA version the installed NVIDIA driver supports, or None."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def choose_build(force_cpu=False):
    if force_cpu or platform.system() == "Darwin":
        return "cpu"
    cuda = driver_cuda_version()
    if cuda is None:
        return "cpu"
    for min_version, tag in CUDA_BUILDS:
        if cuda >= min_version:
            return tag
    return "cpu"   # driver too old for any current CUDA build


def installed_torch():
    """Inspect torch in a separate process so its DLLs aren't locked while reinstalling."""
    code = (
        "import json, torch; print(json.dumps({'version': torch.__version__, "
        "'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def build_of(info):
    """Wheel tag of an installed torch, e.g. '2.12.0+cu130' -> 'cu130'."""
    if info is None:
        return None
    if info["cuda"]:
        return "cu" + info["cuda"].replace(".", "")
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", action="store_true", help="install the CPU-only build")
    ap.add_argument("--check", action="store_true", help="report only, install nothing")
    args = ap.parse_args()

    cuda = driver_cuda_version()
    target = choose_build(args.cpu)
    info = installed_torch()
    current = build_of(info)

    print(f"[torch] NVIDIA driver CUDA version: {'%d.%d' % cuda if cuda else 'none'}")
    print(f"[torch] installed: {info['version'] if info else 'not installed'}")
    print(f"[torch] wanted build: {target}")

    ok = info is not None and current == target and (target == "cpu" or info["available"])
    if ok:
        print("[torch] Correct PyTorch build already installed.")
        return 0
    if args.check:
        print("[torch] Would install the wanted build.")
        return 0

    cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision"]
    if info is not None:
        # same version number with a different +cuXXX/+cpu tag counts as
        # "already satisfied" to pip, so it has to be forced; deps are present
        cmd += ["--force-reinstall", "--no-deps"]
    if target != "cpu":
        cmd += ["--index-url", INDEX.format(target)]
    elif platform.system() == "Linux":
        cmd += ["--index-url", INDEX.format("cpu")]   # PyPI's Linux wheel bundles CUDA
    print("[torch] Running:", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        print("[torch] ERROR: PyTorch installation failed.")
        return 1

    info = installed_torch()
    if info is None:
        print("[torch] ERROR: torch does not import after installation.")
        return 1
    print(f"[torch] Installed {info['version']}, GPU available: {info['available']}")
    if target != "cpu" and not info["available"]:
        print("[torch] WARNING: CUDA build installed but no GPU is usable; S2L will run on the CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
