"""Pair images of the same field across channels (e.g. segment Ph, measure R).

File names like ``..._A6_N_Ph_Processed_<uuid>.tiff`` and
``..._A6_N_R_Processed_<uuid>.tiff`` belong to the same well / stage / type and
differ only in channel (and the random uuid). Tags come from the same parser
the spreadsheet import uses.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from s2l.core.spreadsheet_parser import _extract_tags


def tags(path) -> dict:
    return _extract_tags(Path(path).name)


def channel_of(path) -> str:
    return tags(path).get("channel", "")


def available_channels(paths) -> list[str]:
    return sorted({c for c in (channel_of(p) for p in paths) if c})


def _key(t: dict, with_type: bool = True) -> tuple:
    k = (t.get("well", ""), t.get("timepoint", ""), t.get("stage", ""))
    return k + ((t.get("type", ""),) if with_type else ())


def filter_channel(paths, channel: str | None) -> list[str]:
    """Only the images of *channel* (None / "" = all images)."""
    if not channel:
        return list(paths)
    return [p for p in paths if channel_of(p) == channel]


def build_partner_index(candidates, channel: str):
    """Index the *channel* images by field, for fast lookup."""
    exact, loose = {}, defaultdict(list)
    for c in candidates:
        t = tags(c)
        if t.get("channel") != channel or not t.get("well"):
            continue
        exact.setdefault(_key(t), str(c))
        loose[_key(t, False)].append(str(c))
    return exact, loose


def find_partner(path, index) -> str | None:
    """The *channel* image of the same well / stage / timepoint (same type preferred)."""
    exact, loose = index
    t = tags(path)
    if not t.get("well"):
        return None
    hit = exact.get(_key(t))
    if hit:
        return hit
    alts = loose.get(_key(t, False))
    return alts[0] if alts else None


def pairing_table(paths) -> tuple[list[str], dict[tuple, dict[str, str]]]:
    """(channels, {(well, timepoint, stage, type): {channel: path}}) for a completeness check."""
    chans = available_channels(paths)
    table: dict[tuple, dict[str, str]] = defaultdict(dict)
    for p in paths:
        t = tags(p)
        if t.get("well") and t.get("channel"):
            table[_key(t)][t["channel"]] = str(p)
    return chans, dict(table)
