#!/usr/bin/env python3
# sheet_upsert.py
import os, json, argparse, sys, re
from typing import List, Dict, Tuple

import gspread
from google.oauth2.service_account import Credentials

# --- Your 10 columns (order matters) ---
COLUMNS = [
    "Date",
    "Location",
    "Crust",
    "Dough",
    "Sauce",
    "Cheese",
    "Basil/Extras",
    "Balance/Harmony",
    "Appearance/Aroma",
    "Overall",
    "Tier",
]

# --- Tier mapping with emojis ---
TIER_MAP = {
    "S": "S ⭐️",
    "A": "A 👍",
    "B": "B",
    "C": "C",
    "D": "D ⚠️",
    "E": "E ❌",
    "F": "F ❌ ❌",
}

def _map_tier(value: str) -> str:
    """
    Normalise arbitrary tier strings to one of S/A/B/C/D/E/F,
    then map to the emoji label. If no letter-grade found, return original.
    """
    s = (value or "").strip()
    if not s:
        return ""

    up = s.upper()

    # Prefer a standalone letter (handles "Tier: A", "A", "S tier")
    m = re.search(r"\b([SABCDEF])\b", up)
    if m:
        letter = m.group(1)
    else:
        # Fallback: first allowed letter anywhere
        letter = next((ch for ch in up if ch in "SABCDEF"), None)

    if not letter:
        return s
    return TIER_MAP.get(letter, s)

def load_ws(sheet_id: str, tab_name: str, service_account_file: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(COLUMNS))
    values = ws.get_all_values()
    if not values:
        ws.append_row(COLUMNS, value_input_option="RAW")
    else:
        header = values[0]
        if header != COLUMNS:
            ws.update("1:1", [COLUMNS])
    return ws

def normalise_loc(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def existing_keys(ws) -> set:
    vals = ws.get_all_values()
    if not vals:
        return set()
    idx = {name: i for i, name in enumerate(vals[0])}
    keys = set()
    for row in vals[1:]:
        try:
            d = row[idx["Date"]].strip()
            l = row[idx["Location"]].strip()
        except Exception:
            continue
        if d and l:
            keys.add((d, normalise_loc(l)))
    return keys

def row_from_summary(summary: Dict[str, str]) -> List[str]:
    row = [summary.get(k, "") for k in COLUMNS]
    # Map Tier to emoji label
    try:
        tier_idx = COLUMNS.index("Tier")
        row[tier_idx] = _map_tier(row[tier_idx])
    except Exception:
        pass
    return row

def collect_rows_to_insert(summaries: List[Dict[str, str]], keyset: set) -> List[List[str]]:
    rows = []
    for s in summaries:
        date = (s.get("Date") or "").strip()
        loc  = (s.get("Location") or "").strip()
        if not date or not loc:
            continue
        k = (date, normalise_loc(loc))
        if k in keyset:
            continue
        rows.append(row_from_summary(s))
        keyset.add(k)
    return rows

def upsert_summaries(
    summaries: List[Dict[str, str]],
    sheet_id: str,
    tab_name: str,
    service_account_file: str,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Reusable function you can import:
      from sheet_upsert import upsert_summaries
      inserted, skipped = upsert_summaries(summaries, SHEET_ID, TAB, SA_JSON)

    Returns: (inserted_count, skipped_or_duplicate_count)
    """
    ws = load_ws(sheet_id, tab_name, service_account_file)
    keys = existing_keys(ws)
    to_insert = collect_rows_to_insert(summaries, keys)
    inserted = len(to_insert)
    skipped = len(summaries) - inserted

    if dry_run:
        for r in to_insert:
            print(dict(zip(COLUMNS, r)))
        return inserted, skipped

    if to_insert:
        ws.append_rows(to_insert, value_input_option="USER_ENTERED")
    return inserted, skipped

# ---- CLI ----
def init_upsert_process(json_transcribed_path, sheet_id, sheet_tab, service_acccount_path):
    from pathlib import Path
    dir_path = Path(json_transcribed_path)
    print(dir_path)

    summaries = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                summaries.append(json.load(f))
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)

    if not summaries:
        print("No valid inputs.", file=sys.stderr)
        sys.exit(1)

    ins, skp = upsert_summaries(summaries, sheet_id, sheet_tab, service_acccount_path)
    print(f"Inserted: {ins}  Skipped: {skp}")
