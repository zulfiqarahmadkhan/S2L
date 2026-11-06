"""Load an output folder of per-image ROI sheets into one table, group it and test it."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from s2l.core.image_files import MASK_SUFFIX
from s2l.core.spreadsheet_parser import _extract_tags

NUMERIC_METRICS = ["Area", "Mean Gray Value", "Integrated Density", "Standard Deviation", "Median",
                   "Perimeter", "Feret's diameter", "Modal gray value", "Skewness", "Kurtosis",
                   "Area fraction", "Circularity", "Solidity", "Centroid X", "Centroid Y"]
GROUP_KEYS = ["well", "condition", "stage", "channel", "type", "timepoint", "image"]


def _xy(series):
    out = series.astype(str).str.extract(r"\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)")
    return pd.to_numeric(out[0], errors="coerce"), pd.to_numeric(out[1], errors="coerce")


def _kv(series, key):
    return pd.to_numeric(series.astype(str).str.extract(rf"{key}=([-\d.eE]+)")[0], errors="coerce")


def load_results_folder(folder) -> pd.DataFrame:
    """All per-image sheets (``*_cp_masks.xlsx``) in *folder* as one per-cell table.

    Adds: image, mask_path, well / stage / channel / type / timepoint (from the
    file name), Centroid X/Y, Circularity, Solidity.
    """
    folder = Path(folder)
    frames = []
    for x in sorted(folder.glob(f"*{MASK_SUFFIX}.xlsx")):
        try:
            df = pd.read_excel(x)
        except Exception:
            continue
        if df.empty or "Label" not in df.columns:
            continue
        image = x.stem[: -len(MASK_SUFFIX)]
        df["image"] = image
        df["mask_path"] = str(x.with_suffix(".tif"))
        for k, v in _extract_tags(image).items():
            if k != "tag":
                df[k] = v
        if "Centroid" in df.columns:
            df["Centroid X"], df["Centroid Y"] = _xy(df["Centroid"])
        if "Shape descriptors" in df.columns:
            df["Circularity"] = _kv(df["Shape descriptors"], "circularity")
            df["Solidity"] = _kv(df["Shape descriptors"], "solidity")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for k in ("well", "stage", "channel", "type", "timepoint"):
        if k not in out.columns:
            out[k] = ""
        out[k] = out[k].fillna("").astype(str)
    return out


def input_folder_from_settings(folder) -> str | None:
    """The input folder recorded in the newest run_settings_*.json of an output folder."""
    files = sorted(Path(folder).glob("run_settings_*.json"))
    for f in reversed(files):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("input_folder") and Path(d["input_folder"]).is_dir():
                return d["input_folder"]
        except (ValueError, OSError):
            continue
    return None


def parse_condition_map(text: str) -> dict[str, str]:
    """``"A6,A7=WT; A8-A10=mutant"`` → {well: condition}. Ranges work within one row letter."""
    mapping = {}
    for part in re.split(r"[;\n]", text or ""):
        if "=" not in part:
            continue
        wells, name = part.split("=", 1)
        name = name.strip()
        for tok in wells.split(","):
            tok = tok.strip().upper()
            m = re.fullmatch(r"([A-H])(\d{1,2})\s*-\s*([A-H])?(\d{1,2})", tok)
            if m:
                row, a, b = m.group(1), int(m.group(2)), int(m.group(4))
                for n in range(min(a, b), max(a, b) + 1):
                    mapping[f"{row}{n}"] = name
            elif tok:
                mapping[tok] = name
    return mapping


def add_condition(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = df["well"].map(mapping).fillna("(unassigned)") if mapping else df["well"]
    return df


def group_table(df: pd.DataFrame, metric: str, by: str) -> pd.DataFrame:
    """Per group: cells, images, mean, SD, median, and mean ± SD of the image means."""
    d = df.dropna(subset=[metric])
    per_image = d.groupby(list(dict.fromkeys([by, "image"])))[metric].mean().reset_index()
    t = d.groupby(by)[metric].agg(cells="count", mean="mean", sd="std", median="median")
    t["images"] = per_image.groupby(by)["image"].count()
    t["image_mean"] = per_image.groupby(by)[metric].mean()
    t["image_sd"] = per_image.groupby(by)[metric].std()
    return t.reset_index()[[by, "images", "cells", "mean", "sd", "median", "image_mean", "image_sd"]]


def compare(df: pd.DataFrame, metric: str, by: str, unit: str = "cells") -> list[str]:
    """Statistical comparison of *metric* between groups. Returns readable lines.

    unit="cells" treats every cell as a sample (many cells, but cells of one
    image are not independent); unit="images" compares image means (honest
    when you have several images per group).
    """
    from scipy import stats
    d = df.dropna(subset=[metric])
    if unit == "images":
        d = d.groupby(list(dict.fromkeys([by, "image"])))[metric].mean().reset_index()
    groups = {g: v[metric].to_numpy() for g, v in d.groupby(by)}
    groups = {g: v for g, v in groups.items() if len(v) > 0}
    names = list(groups)
    lines = []
    if len(names) < 2:
        return ["Need at least two groups to compare."]
    small = [g for g in names if len(groups[g]) < 3]
    if small:
        lines.append(f"Too few samples for a reliable test in: {', '.join(map(str, small))} (need ≥ 3 {unit}).")
    if len(names) == 2:
        a, b = groups[names[0]], groups[names[1]]
        if len(a) >= 2 and len(b) >= 2:
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            t = stats.ttest_ind(a, b, equal_var=False)
            lines.append(f"{names[0]} vs {names[1]} ({unit}): Mann-Whitney p = {u.pvalue:.3g}; "
                         f"Welch t-test p = {t.pvalue:.3g}")
    else:
        valid = [groups[g] for g in names if len(groups[g]) >= 2]
        if len(valid) >= 2:
            k = stats.kruskal(*valid)
            lines.append(f"Kruskal-Wallis across {len(names)} groups ({unit}): p = {k.pvalue:.3g}")
        pairs, ps = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = groups[names[i]], groups[names[j]]
                if len(a) >= 2 and len(b) >= 2:
                    pairs.append((names[i], names[j]))
                    ps.append(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        if ps:
            # Holm-Bonferroni correction for the pairwise tests
            order = np.argsort(ps)
            adj = np.empty(len(ps))
            running = 0.0
            for rank, idx in enumerate(order):
                running = max(running, min(1.0, (len(ps) - rank) * ps[idx]))
                adj[idx] = running
            lines.append("Pairwise Mann-Whitney (Holm-corrected):")
            for (g1, g2), p_adj in zip(pairs, adj):
                star = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
                lines.append(f"   {g1} vs {g2}: p = {p_adj:.3g} {star}")
    if unit == "cells":
        lines.append("Note: cells from the same image are not independent; with several images per group, "
                     "prefer “image means” as the statistical unit.")
    return lines
