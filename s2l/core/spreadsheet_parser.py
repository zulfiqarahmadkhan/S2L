"""Parse experiment spreadsheets (xlsx) to extract image file paths and tags.

Supports the FIM sheet format where:
  - Well IDs (A1, A2, ...) appear as column headers
  - Rows contain cycle/timepoint data with 'Image' markers
  - Actual .tiff files live in an Images/<well>/ directory tree
"""
import re
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TIFF_EXTENSIONS = {".tif", ".tiff"}

# Regex to extract tags from filenames like:
#   ..._A1_1_O_B_Raw_<uuid>.tiff
#   ..._A1_1_AN_Bf_Processed_<uuid>.tiff
_TAG_RE = re.compile(
    r"_([A-H]\d+)_(?:(\d+)_)?"     # well + optional timepoint
    r"(O|N|AN)_"                     # stage
    r"(BfRGB|Bf|B|R|PhRGB|Ph)_"     # channel
    r"(Raw|Processed)_"              # type
)

# Well ID pattern
_WELL_RE = re.compile(r"^[A-H]\d{1,2}$")


def _find_images_root(xlsx_path: Path) -> Path | None:
    """Walk up from the xlsx to find an 'Images' directory nearby."""
    for parent in [xlsx_path.parent, xlsx_path.parent.parent, xlsx_path.parent.parent.parent]:
        candidate = parent / "Images"
        if candidate.is_dir():
            return candidate
    return None


def parse_fim_sheet(xlsx_path: str, sheet_name: str | None = None) -> list[dict]:
    """Parse an experiment xlsx and return image records.

    Each record: {path, well, timepoint, stage, channel, type, tag}
    """
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        raise FileNotFoundError(f"Spreadsheet not found: {xlsx_path}")

    logger.info(f"Parsing {xlsx.name}")

    # Auto-detect sheet name
    xls = pd.ExcelFile(str(xlsx))
    if sheet_name and sheet_name in xls.sheet_names:
        target_sheet = sheet_name
    else:
        # Try common names
        for candidate in ["FIM sheet", "FIM", "Sheet1"]:
            if candidate in xls.sheet_names:
                target_sheet = candidate
                break
        else:
            target_sheet = xls.sheet_names[0]
    logger.info(f"Using sheet: '{target_sheet}'")

    df = pd.read_excel(str(xlsx), sheet_name=target_sheet, header=None)

    # Find the well columns: scan for a row where cells match well ID pattern (A1, A2, B1, etc.)
    wells = {}  # col_index -> well_id
    for row_idx in range(min(50, len(df))):
        for col_idx in range(len(df.columns)):
            val = str(df.iloc[row_idx, col_idx]).strip() if pd.notna(df.iloc[row_idx, col_idx]) else ""
            if _WELL_RE.match(val):
                wells[col_idx] = val
        if wells:
            logger.info(f"Found well headers at row {row_idx}: {list(wells.values())}")
            break

    if not wells:
        logger.warning("No well IDs found in spreadsheet, falling back to directory scan")
        return _fallback_directory_scan(xlsx)

    # Find the Images directory
    images_root = _find_images_root(xlsx)
    if not images_root:
        raise FileNotFoundError(
            f"Could not find 'Images' directory near {xlsx_path}. "
            "Expected structure: .../Export/xlsx/file.xlsx with .../Images/ nearby."
        )
    logger.info(f"Images root: {images_root}")

    # Scan all .tiff files from the Images directory, organized by well
    all_tiffs: dict[str, list[Path]] = {}  # well -> list of paths
    flat_by_well = None  # lazily built when files are not in per-well folders
    for well_id in sorted(set(wells.values())):
        well_dir = images_root / well_id
        if well_dir.is_dir():
            tiffs = sorted(
                p for p in well_dir.rglob("*")
                if p.suffix.lower() in TIFF_EXTENSIONS
            )
        else:
            # Flat layout: Images/<prefix>_<well>_..._<uuid>.tiff
            if flat_by_well is None:
                flat_by_well = {}
                for p in sorted(images_root.glob("*")):
                    if p.suffix.lower() in TIFF_EXTENSIONS:
                        flat_by_well.setdefault(_extract_tags(p.name)["well"], []).append(p)
            tiffs = flat_by_well.get(well_id, [])
        all_tiffs[well_id] = tiffs
        logger.info(f"  {well_id}: {len(tiffs)} .tiff files")

    # Build records with tags
    records = []
    for well_id, paths in all_tiffs.items():
        for p in paths:
            rec = _extract_tags(p.name)
            rec.update({"path": str(p.resolve()), "well": well_id})
            records.append(rec)

    logger.info(f"Total: {len(records)} .tiff records from {len(all_tiffs)} wells")
    return records


def _fallback_directory_scan(xlsx: Path) -> list[dict]:
    """If no well structure found, just scan for all .tiff files near the xlsx."""
    images_root = _find_images_root(xlsx)
    if not images_root:
        return []
    records = []
    for p in sorted(images_root.rglob("*")):
        if p.suffix.lower() in TIFF_EXTENSIONS:
            rec = {"path": str(p.resolve())}
            rec.update(_extract_tags(p.name))
            records.append(rec)
    return records


def _extract_tags(filename: str) -> dict:
    """Extract well, timepoint, stage, channel, type from a filename."""
    m = _TAG_RE.search(filename)
    if m:
        well, tp, stage, channel, ftype = m.groups()
        return {
            "well": well,
            "timepoint": tp or "",
            "stage": stage,
            "channel": channel,
            "type": ftype,
            "tag": f"{stage}_{channel}_{ftype}",
        }

    # Flexible fallback matching if filename doesn't match strict _TAG_RE
    well, tp, stage, channel, ftype = "", "", "", "", ""

    # Well pattern (e.g. A1, A01, B12)
    m_well = re.search(r"(?:_|^|\b)([A-H])0?(\d{1,2})(?:_|$|\b)", filename, re.IGNORECASE)
    if m_well:
        well = f"{m_well.group(1).upper()}{int(m_well.group(2))}"

    # Stage pattern (O, N, AN, Baseline, Treatment, etc.)
    m_stage = re.search(r"(?:_|^|\b)(O|N|AN|Baseline|Control|Treatment|Stage\d*)(?:_|$|\b)", filename, re.IGNORECASE)
    if m_stage:
        stage = m_stage.group(1).upper()

    # Channel pattern (BfRGB, Bf, B, R, PhRGB, Ph, GFP, DAPI, RFP, Cy5, FITC, Red, Green, Blue, Phase, Brightfield)
    m_chan = re.search(r"(?:_|^|\b)(BfRGB|Bf|B|R|PhRGB|Ph|GFP|DAPI|RFP|Cy5|FITC|Red|Green|Blue|Phase|Brightfield)(?:_|$|\b)", filename, re.IGNORECASE)
    if m_chan:
        channel = m_chan.group(1)

    # Type pattern (Raw, Processed, Proc, Img, Image)
    m_type = re.search(r"(?:_|^|\b)(Raw|Processed|Proc|Img|Image)(?:_|$|\b)", filename, re.IGNORECASE)
    if m_type:
        ftype = m_type.group(1).capitalize()

    tag = "_".join(x for x in [stage, channel, ftype] if x)
    return {
        "well": well,
        "timepoint": tp,
        "stage": stage,
        "channel": channel,
        "type": ftype,
        "tag": tag,
    }


def get_unique_tags(records: list[dict]) -> list[str]:
    return sorted({r["tag"] for r in records if r["tag"]})


def filter_records(records: list[dict], active_tags: set[str]) -> list[dict]:
    if not active_tags:
        return records
    return [r for r in records if r["tag"] in active_tags]
