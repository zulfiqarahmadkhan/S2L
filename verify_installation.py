#!/usr/bin/env python3
"""
Installation verification script for S2L.
Checks system dependencies, PyTorch/CUDA availability, Cellpose, StarDist, and SAM compatibility.
"""
import sys
import os

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'Installed')
        return True, version
    except ImportError:
        return False, "Not Installed"

def run_verification():
    print("=" * 60)
    print("S2L Installation Verification")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print("-" * 60)

    packages = [
        ("PyQt6", "PyQt6"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("OpenPyXL", "openpyxl"),
        ("XlsxWriter", "xlsxwriter"),
        ("Matplotlib", "matplotlib"),
        ("OpenCV", "cv2"),
        ("Pillow (PIL)", "PIL"),
        ("Cellpose", "cellpose"),
        ("StarDist", "stardist"),
        ("PyTorch", "torch"),
    ]

    all_ok = True
    print("\n📦 Checking Installed Packages:")
    for label, pkg in packages:
        ok, ver = check_package(label, pkg)
        status = f"✅ {ver}" if ok else "❌ Missing"
        print(f"  - {label:<15}: {status}")
        if not ok and label not in ["StarDist"]: # StarDist is optional
            all_ok = False

    print("\n⚡ Checking Hardware Acceleration (CUDA):")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  - PyTorch CUDA    : ✅ Available ({gpu_name})")
        else:
            print("  - PyTorch CUDA    : ⚠️ Not Available (CPU mode)")
    except Exception as e:
        print(f"  - PyTorch CUDA    : ❌ Error checking CUDA ({e})")

    print("\n🤖 Checking SAM Compatibility:")
    try:
        from s2l.utils.sam_utils import check_sam_compatibility
        sam_ok, sam_msg = check_sam_compatibility()
        sam_status = f"✅ {sam_msg}" if sam_ok else f"⚠️ {sam_msg}"
        print(f"  - Cellpose-SAM    : {sam_status}")
    except Exception as e:
        print(f"  - Cellpose-SAM    : ❌ Error checking SAM ({e})")

    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 Verification Complete: All core S2L dependencies are installed.")
    else:
        print("⚠️ Verification Complete with Warnings: Some dependencies are missing.")
    print("=" * 60)
    return all_ok

if __name__ == "__main__":
    # emoji would crash a cp1252 console or a redirected output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = run_verification()
    # Keep console open if run by double clicking (not when output is captured)
    if os.name == 'nt' and sys.stdin is not None and sys.stdin.isatty() and 'PYTHONUNBUFFERED' not in os.environ:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
    sys.exit(0 if ok else 1)
