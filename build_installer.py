#!/usr/bin/env python3
"""Build the S2L installers from the current code.

    python build_installer.py

Produces, in the project folder:
  S2L_Setup.exe   Windows: copies S2L to %USERPROFILE%\\S2L, adds a desktop shortcut and
                  runs install_and_run_local.bat (installs Miniforge/conda if needed,
                  the S2L environment, PyTorch for the GPU, packages, models) and starts S2L.
                  Compiled with the C# compiler that ships with Windows (.NET Framework 4).
  S2L_mac.zip     macOS: S2L.app — copies S2L to ~/S2L and runs install_and_run_local.sh
                  in Terminal (same steps). Built here without a Mac, so it is not signed:
                  see "READ ME FIRST (Mac).txt" inside the zip for opening it the first time.
"""
import base64
import io
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PAYLOAD_FILES = ["main.py", "requirements.txt", "requirements-optional.txt", "install_torch.py",
                 "verify_installation.py", "README.md", "app_icon.ico",
                 "install_and_run_local.bat", "install_and_run_local.sh"]
PAYLOAD_DIRS = ["s2l", "docs"]
EXECUTABLE = {"install_and_run_local.sh", "S2L.command"}


def payload_files():
    """(path on disk, path inside the package) for every file S2L needs at runtime."""
    for name in PAYLOAD_FILES:
        p = ROOT / name
        if not p.exists():
            raise SystemExit(f"Missing {name}")
        yield p, name
    for d in PAYLOAD_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                yield p, p.relative_to(ROOT).as_posix()


# ═══════════════════════════════════════════════════════════════════════════
# Windows
# ═══════════════════════════════════════════════════════════════════════════

CS_TEMPLATE = r"""using System;
using System.IO;
using System.IO.Compression;
using System.Diagnostics;

class Program {
    static void Main() {
        try {
            Console.WriteLine("========================================");
            Console.WriteLine("            S2L installer");
            Console.WriteLine("========================================");
            Console.WriteLine();
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string target = Path.Combine(home, "S2L");
            Directory.CreateDirectory(target);
            Console.WriteLine("[*] Installing S2L to: " + target);

            byte[] zipBytes = Convert.FromBase64String("@@PAYLOAD@@");
            using (var ms = new MemoryStream(zipBytes))
            using (var archive = new ZipArchive(ms, ZipArchiveMode.Read)) {
                foreach (ZipArchiveEntry entry in archive.Entries) {
                    if (string.IsNullOrEmpty(entry.Name)) continue;
                    string dest = Path.GetFullPath(Path.Combine(target, entry.FullName.Replace('/', '\\')));
                    if (!dest.StartsWith(target, StringComparison.OrdinalIgnoreCase)) continue;
                    Directory.CreateDirectory(Path.GetDirectoryName(dest));
                    entry.ExtractToFile(dest, true);
                }
            }
            Console.WriteLine("[*] Files copied.");

            string bat = Path.Combine(target, "install_and_run_local.bat");
            string icon = Path.Combine(target, "app_icon.ico");
            try {
                string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
                string lnk = Path.Combine(desktop, "S2L.lnk");
                string ps = "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('" + lnk.Replace("'", "''") + "');" +
                            "$s.TargetPath='" + bat.Replace("'", "''") + "';" +
                            "$s.WorkingDirectory='" + target.Replace("'", "''") + "';" +
                            "$s.IconLocation='" + icon.Replace("'", "''") + "';" +
                            "$s.Description='S2L cell segmentation';$s.Save()";
                var sp = new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -Command \"" + ps + "\"");
                sp.UseShellExecute = false;
                sp.CreateNoWindow = true;
                Process.Start(sp).WaitForExit(20000);
                Console.WriteLine("[*] Desktop shortcut 'S2L' created (use it to start S2L next time).");
            } catch (Exception) {
                Console.WriteLine("[WARN] Could not create the desktop shortcut.");
            }

            Console.WriteLine("[*] Starting setup (the first run downloads several GB and takes a while)...");
            Console.WriteLine();
            var psi = new ProcessStartInfo("cmd.exe", "/c \"\"" + bat + "\"\"");
            psi.WorkingDirectory = target;
            psi.UseShellExecute = true;
            Process.Start(psi);
        } catch (Exception ex) {
            Console.WriteLine("[ERROR] Installation failed: " + ex.Message);
            Console.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }
}
"""


def build_windows():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in payload_files():
            zf.write(src, arc)
    cs = CS_TEMPLATE.replace("@@PAYLOAD@@", base64.b64encode(buf.getvalue()).decode())
    cs_file = ROOT / "_installer_tmp.cs"
    cs_file.write_text(cs, encoding="utf-8")
    exe = ROOT / "S2L_Setup.exe"
    csc = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
    if not csc.exists():
        cs_file.unlink()
        print("[Windows] skipped: the C# compiler (.NET Framework 4) is only available on Windows.")
        return
    try:
        res = subprocess.run([str(csc), "/nologo", f"/out:{exe}", "/target:exe",
                              f"/win32icon:{ROOT / 'app_icon.ico'}",
                              "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll", str(cs_file)], capture_output=True, text=True)
    finally:
        cs_file.unlink()
    if res.returncode != 0:
        raise SystemExit("[Windows] compilation failed:\n" + (res.stdout or res.stderr))
    print(f"[Windows] {exe}  ({exe.stat().st_size / 1e6:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════
# macOS
# ═══════════════════════════════════════════════════════════════════════════

MAC_LAUNCHER = """#!/bin/bash
# S2L.app: copy the program to ~/S2L and run its setup + launcher in Terminal.
SRC="$(cd "$(dirname "$0")/../Resources/payload" && pwd)"
DEST="$HOME/S2L"
mkdir -p "$DEST"
/usr/bin/rsync -a "$SRC/" "$DEST/" 2>/dev/null || cp -R "$SRC/." "$DEST/"
chmod +x "$DEST/install_and_run_local.sh" "$DEST/S2L.command"
# the user already approved this app; don't let macOS ask again for every copied file
/usr/bin/xattr -dr com.apple.quarantine "$DEST" 2>/dev/null
open -a Terminal "$DEST/S2L.command"
"""

MAC_COMMAND = """#!/bin/bash
cd "$(dirname "$0")"
exec bash ./install_and_run_local.sh
"""

INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>S2L</string>
  <key>CFBundleDisplayName</key><string>S2L</string>
  <key>CFBundleIdentifier</key><string>org.s2l.launcher</string>
  <key>CFBundleVersion</key><string>@@VERSION@@</string>
  <key>CFBundleShortVersionString</key><string>@@VERSION@@</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>S2L</string>
  <key>CFBundleIconFile</key><string>S2L</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""

MAC_README = """S2L for macOS
=============

1. Unzip this file (double-click) and drag S2L.app to your Applications folder.
2. Open S2L.app. The first time, macOS blocks it because it is not from the App Store:
     • macOS 15 (Sequoia) and newer: click "Done", then open System Settings →
       Privacy & Security, scroll down and click "Open Anyway" next to S2L, and confirm.
     • macOS 14 and older: right-click (or Control-click) S2L.app → Open → Open.
   You only need to do this once.
   If macOS says the app "is damaged", open Terminal and run:
       xattr -dr com.apple.quarantine /Applications/S2L.app
3. A Terminal window opens and sets everything up. The first run downloads several GB
   (conda, PyTorch, the AI models) and takes 10-30 minutes; later starts take seconds.
   Keep the Terminal window open while you use S2L.

S2L is installed to ~/S2L (your home folder). It uses the Apple GPU (Metal) for Cellpose
on Apple-silicon Macs. StarDist needs TensorFlow, which is not available for every Mac;
if it cannot be installed, S2L still works and only StarDist is unavailable.
"""


def _zip_add(zf, arcname, data: bytes, executable=False):
    info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
    info.create_system = 3                       # Unix, so macOS keeps the permission bits
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


def _icns() -> bytes | None:
    try:
        from PIL import Image
        img = Image.open(ROOT / "app_icon.ico")
        img = img.convert("RGBA").resize((512, 512), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="ICNS")
        return out.getvalue()
    except Exception as e:
        print(f"[macOS] no app icon ({e})")
        return None


def build_mac():
    from s2l import __version__
    out = ROOT / "S2L_mac.zip"
    app = "S2L.app/Contents"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        _zip_add(zf, f"{app}/Info.plist", INFO_PLIST.replace("@@VERSION@@", __version__).encode())
        _zip_add(zf, f"{app}/MacOS/S2L", MAC_LAUNCHER.encode(), executable=True)
        icns = _icns()
        if icns:
            _zip_add(zf, f"{app}/Resources/S2L.icns", icns)
        for src, arc in payload_files():
            data = src.read_bytes()
            if arc.endswith(".sh"):
                data = data.replace(b"\r\n", b"\n")      # bash on macOS rejects Windows line endings
            _zip_add(zf, f"{app}/Resources/payload/{arc}", data, executable=Path(arc).name in EXECUTABLE)
        _zip_add(zf, f"{app}/Resources/payload/S2L.command", MAC_COMMAND.encode(), executable=True)
        _zip_add(zf, "READ ME FIRST (Mac).txt", MAC_README.encode())
    print(f"[macOS]   {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    print("Building S2L installers...")
    build_windows()
    build_mac()
    print("Done.")
