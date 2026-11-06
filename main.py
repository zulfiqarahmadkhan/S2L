"""Entry point for S2L."""
import sys
import os
import traceback
import platform

# Write crashes to a log file so PyInstaller failures are visible
def _crash_log(exc_type, exc_val, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    try:
        log_path = os.path.join(os.path.dirname(sys.executable), "s2l_crash.log")
        with open(log_path, "w") as f:
            f.write(msg)
    except Exception:
        pass
    # Also try writing next to the script
    try:
        log_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2l_crash.log")
        with open(log_path2, "w") as f:
            f.write(msg)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_val, exc_tb)

sys.excepthook = _crash_log

# Fix PyTorch DLL loading issue on Windows (PyInstaller + PyQt6 conflict)
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

from s2l.ui.main_window import main

if __name__ == "__main__":
    main()
