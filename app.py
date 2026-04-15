import re
import io
import json
import time
import uuid
import base64
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

APP_TITLE = "CT IEC Constancy QC Reporter"

# =========================================================
# EDIT THESE ONCE
# =========================================================
DEFAULT_GITHUB_OWNER = "YOUR_GITHUB_USERNAME"
DEFAULT_GITHUB_REPO = "YOUR_REPO_NAME"
DEFAULT_GITHUB_BRANCH = "main"
DEFAULT_GITHUB_CSV_PATH = "ct_qc_data/ct_qc_history.csv"

DATA_DIR = Path("ct_qc_data")
LOCAL_HISTORY_CSV = DATA_DIR / "ct_qc_history.csv"
LOCAL_LOCK_FILE = DATA_DIR / "ct_qc_history.lock"
REPORTS_DIR = DATA_DIR / "reports"
CHARTS_DIR = DATA_DIR / "charts"
LOGO_PATH = DATA_DIR / "logo.png"  # optional

DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
NUM = r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)"
TOL_SEP = r"(?:…|\.\.\.|\u2026|\u202f…\u202f|\s+…\s+|\s+\.\.\.\s+)"

SECTION_PATTERNS = {
    "homogeneity_start": r"(?m)^1\s+Homogeneity\s*\(IEC\s*Constancy\)\s*$",
    "noise_start": r"(?m)^2\s+Noise\s*\(IEC\s*Constancy\)\s*$",
    "mtf_start": r"(?m)^3\s+MTF\s*\(IEC\s*Constancy\)\s*$",
    "table_start": r"(?m)^4\s+Table\s+Positioning\s*\(IEC\s*Constancy\)\s*$",
    "tube_start": r"(?m)^5\s+Tube\s+Voltage\s*\(IEC\s*Constancy\)\s*$",
    "image_start": r"(?m)^6\s+Image\s+Inspection\s*\(Constancy\)\s*$",
}

MODE_HEADER_PATTERNS = {
    "homogeneity": r"(?m)^1\.3\.(\d+)\s+(.+)$",
    "noise": r"(?m)^2\.3\.(\d+)\s+(.+)$",
    "mtf": r"(?m)^3\.3\.(\d+)\s+(.+)$",
}

DEBUG_MODE = True


def normalize_pdf_text(text):
    return (
        str(text)
        .replace("\u202f", " ")
        .replace("\xa0", " ")
        .replace("…", "...")
    )


def safe_float(text):
    try:
        return float(str(text).replace(",", ".").strip())
    except Exception:
        return None


def validate_iso_timestamp(ts: str) -> bool:
    try:
        datetime.fromisoformat(ts)
        return True
    except Exception:
        return False


def build_scanner_id(site_name: str, scanner_name: str) -> str:
    raw = f"{str(site_name).strip()}__{str(scanner_name).strip()}".lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw or "unknown_scanner"


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()) or "file"


def normalize_ws(text: str) -> str:
    return "\n".join(" ".join(line.replace("\u202f", " ").split()) for line in str(text).splitlines())


def compact_mode_name(name: str) -> str:
    name = " ".join(str(name).split())
    name = re.sub(r"\s+\|\s+Serial Number:.*$", "", name, flags=re.I)
    return name.strip()


def format_num(v, digits=3):
    if v is None or pd.isna(v):
        return ""
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if abs(fv - round(fv)) < 1e-9:
        return str(int(round(fv)))
    return f"{fv:.{digits}f}"


def get_history_columns():
    return [
        "timestamp",
        "session_label",
        "site_name",
        "scanner_name",
        "scanner_id",
        "test_name",
        "value",
        "unit",
        "criteria",
        "status",
        "details",
        "source_file",
        "sequence_label",
    ]


def empty_history_df():
    return pd.DataFrame(columns=get_history_columns())


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_history_df()

    df = df.copy()

    for col in get_history_columns():
        if col not in df.columns:
            df[col] = None

    text_cols = [
        "timestamp",
        "session_label",
        "site_name",
        "scanner_name",
        "scanner_id",
        "test_name",
        "unit",
        "criteria",
        "status",
        "details",
        "source_file",
        "sequence_label",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    missing_id = df["scanner_id"].str.strip() == ""
    if missing_id.any():
        df.loc[missing_id, "scanner_id"] = df.loc[missing_id].apply(
            lambda r: build_scanner_id(r["site_name"], r["scanner_name"]),
            axis=1,
        )

    return df[get_history_columns()]


def github_is_ready(cfg):
    return bool(
        cfg
        and cfg.get("token")
        and cfg.get("owner")
        and cfg.get("repo")
        and cfg.get("path")
        and cfg["owner"] != "YOUR_GITHUB_USERNAME"
        and cfg["repo"] != "YOUR_REPO_NAME"
    )


def get_ct_test_order():
    return [
        "Water Value",
        "Homogeneity",
        "Noise",
        "MTF 50%",
        "MTF 10%",
        "Table Positioning (Continuous)",
        "Table Positioning (Stepwise)",
        "Tube Voltage",
        "Image Inspection",
    ]


def ct_sort_key(test_name):
    order = get_ct_test_order()
    if test_name in order:
        return (order.index(test_name), str(test_name))
    return (999, str(test_name))


def sort_tests_ct(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "test_name" not in df.columns:
        return df

    out = df.copy()
    out["_ct_order"] = out["test_name"].apply(ct_sort_key)

    preferred_cols = ["_ct_order", "sequence_label", "timestamp"]
    sort_cols = [c for c in preferred_cols if c in out.columns]

    out = out.sort_values(sort_cols, na_position="last").drop(columns=["_ct_order"])
    return out


def build_single_session_df(history_df, scanner_id, timestamp):
    df = normalize_history_df(history_df).copy()
    if df.empty:
        return empty_history_df()

    out = df[
        (df["scanner_id"].astype(str) == str(scanner_id))
        & (df["timestamp"].astype(str) == str(timestamp))
    ].copy()
    return sort_tests_ct(out)


def read_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    uploaded_file.seek(0)
    return "\n\n".join(pages)


def extract_pdf_metadata(text: str):
    raw = str(text)
    clean = normalize_ws(text)

    site_name = ""
    scanner_name = ""
    serial_number = ""

    m_head_product = re.search(
        r"^(.*?)\nIEC Constancy\nSerial Number:\s*([A-Za-z0-9-]+)\n(.+)$",
        clean,
        re.M,
    )
    if m_head_product:
        scanner_name = m_head_product.group(1).strip()
        serial_number = m_head_product.group(2).strip()
        site_name = m_head_product.group(3).strip()

    m_prod = re.search(r"^Product Name\s+(.+?)\s*$", raw, re.I | re.M)
    if m_prod:
        scanner_name = " ".join(m_prod.group(1).split())

    m_serial = re.search(r"^Serial Number\s+(.+?)\s*$", raw, re.I | re.M)
    if m_serial:
        serial_number = " ".join(m_serial.group(1).split())

    m_hosp = re.search(r"^Hospital Name\s+(.+?)\s*$", raw, re.I | re.M)
    if m_hosp:
        site_name = " ".join(m_hosp.group(1).split())

    if not site_name:
        m_footer = re.search(
            r"([A-Z][A-Z0-9 .&/\-]+)\s+\|\s+([A-Za-z0-9._\- ]+)\s+\|\s+Serial Number:\s*([A-Za-z0-9\-]+)",
            clean,
            re.I,
        )
        if m_footer:
            site_name = m_footer.group(1).strip()
            scanner_name = scanner_name or m_footer.group(2).strip()
            serial_number = serial_number or m_footer.group(3).strip()

    m_ts = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M)", clean, re.I)
    timestamp_iso = ""
    if m_ts:
        try:
            dt = datetime.strptime(m_ts.group(1), "%m/%d/%Y %I:%M:%S %p")
            timestamp_iso = dt.isoformat(timespec="seconds")
        except Exception:
            timestamp_iso = ""

    return {
        "site_name": site_name,
        "scanner_name": scanner_name,
        "serial_number": serial_number,
        "timestamp_iso": timestamp_iso,
    }


def extract_section(text, start_pattern, end_pattern=None):
    text = normalize_pdf_text(text)
    starts = list(re.finditer(start_pattern, text, flags=re.I | re.M))
    if not starts:
        return ""

    start_match = starts[-1]
    start_idx = start_match.start()

    if end_pattern:
        ends = list(re.finditer(end_pattern, text[start_idx + 1 :], flags=re.I | re.M))
        if ends:
            end_idx = start_idx + 1 + ends[0].start()
            return text[start_idx:end_idx]

    return text[start_idx:]


def iter_clean_lines(section):
    for raw in str(section).splitlines():
        line = " ".join(raw.replace("\u202f", " ").split())
        if line:
            yield line


def value_in_range(value, low, high):
    if value is None or low is None or high is None:
        return False
    return low <= value <= high


def split_mode_blocks(section_text: str, header_pattern: str, name_group: int = 2):
    section_text = normalize_pdf_text(section_text)
    matches = list(re.finditer(header_pattern, section_text, flags=re.M))
    if not matches:
        return []

    blocks = []
    for i, m in enumerate(matches):
        name = compact_mode_name(m.group(name_group))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        block_text = section_text[start:end]
        blocks.append((name, block_text))
    return blocks


def parse_tolerance_line(line):
    line = normalize_pdf_text(line)
    m = re.search(rf"Tolerance:\s*({NUM})\s*{TOL_SEP}\s*({NUM})", line, re.I)
    if m:
        return safe_float(m.group(1)), safe_float(m.group(2))

    nums = [safe_float(x) for x in re.findall(NUM, line)]
    if ("Tolerance" in line or "Reference Tolerance" in line) and len(nums) >= 2 and "..." in line:
        return nums[-2], nums[-1]

    return None, None


def row_status(value, low, high):
    return "PASS" if value_in_range(value, low, high) else "FAIL"


def summarize_slice_rows(rows, metric_name, worst_by="max_abs"):
    if not rows:
        return None

    if worst_by == "max_abs":
        worst = max(rows, key=lambda x: abs(x["value"]))
    elif worst_by == "max":
        worst = max(rows, key=lambda x: x["value"])
    elif worst_by == "min":
        worst = min(rows, key=lambda x: x["value"])
    else:
        worst = rows[0]

    overall_status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    return worst, overall_status


def make_result(test_name, sequence_label, value, unit, criteria, status, details):
    return {
        "test_name": test_name,
        "value": value,
        "unit": unit,
        "criteria": criteria,
        "status": status,
        "details": details,
        "sequence_label": sequence_label,
    }


def debug_dump_sections(pdf_text):
    if not DEBUG_MODE:
        return

    sections = {
        "HOMOGENEITY": SECTION_PATTERNS["homogeneity_start"],
        "NOISE": SECTION_PATTERNS["noise_start"],
        "MTF": SECTION_PATTERNS["mtf_start"],
        "TABLE": SECTION_PATTERNS["table_start"],
        "TUBE": SECTION_PATTERNS["tube_start"],
        "IMAGE": SECTION_PATTERNS["image_start"],
    }

    with st.expander("DEBUG: section match dump"):
        for name, pattern in sections.items():
            matches = list(re.finditer(pattern, pdf_text, flags=re.I | re.M))
            st.write(f"### {name}")
            st.write(f"Matches found: {len(matches)}")
            for i, m in enumerate(matches):
                start = max(0, m.start() - 80)
                end = min(len(pdf_text), m.end() + 200)
                snippet = pdf_text[start:end]
                st.code(f"[Match {i}]\n{snippet}")


# =========================================================
# CT PARSERS
# =========================================================
def parse_ct_water_value_and_homogeneity(text):
    section = extract_section(
        text,
        SECTION_PATTERNS["homogeneity_start"],
        SECTION_PATTERNS["noise_start"],
    )
    mode_blocks = split_mode_blocks(section, MODE_HEADER_PATTERNS["homogeneity"])

    results = []
    for mode_name, block in mode_blocks:
        lines = list(iter_clean_lines(block))

        # -------- Water Value --------
        water_rows = []
        current_low, current_high = None, None
        in_water = False

        for line in lines:
            if "Water Value Results" in line:
                in_water = True
                continue

            if re.match(r"^Slice\s+\d+$", line, re.I):
                in_water = False

            low, high = parse_tolerance_line(line)
            if low is not None and high is not None:
                current_low, current_high = low, high
                continue

            if not in_water:
                continue

            m_inline = re.search(
                rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})$",
                line,
                re.I,
            )
            if m_inline:
                water_rows.append(
                    {
                        "slice": int(m_inline.group(1)),
                        "value": safe_float(m_inline.group(2)),
                        "reference": safe_float(m_inline.group(3)),
                        "low": safe_float(m_inline.group(4)),
                        "high": safe_float(m_inline.group(5)),
                    }
                )
                water_rows[-1]["status"] = row_status(water_rows[-1]["value"], water_rows[-1]["low"], water_rows[-1]["high"])
                continue

            m_simple = re.search(rf"^(\d+)\s+({NUM})\s+({NUM})$", line, re.I)
            if m_simple and current_low is not None and current_high is not None:
                water_rows.append(
                    {
                        "slice": int(m_simple.group(1)),
                        "value": safe_float(m_simple.group(2)),
                        "reference": safe_float(m_simple.group(3)),
                        "low": current_low,
                        "high": current_high,
                    }
                )
                water_rows[-1]["status"] = row_status(water_rows[-1]["value"], water_rows[-1]["low"], water_rows[-1]["high"])

        if water_rows:
            worst, overall_status = summarize_slice_rows(water_rows, "Water Value", worst_by="max_abs")
            slice_summary = "; ".join(
                f"S{r['slice']}={format_num(r['value'])} HU [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
                for r in water_rows
            )
            results.append(
                make_result(
                    "Water Value",
                    mode_name,
                    abs(worst["value"]),
                    "HU",
                    "Worst absolute central ROI CT number within tolerance",
                    overall_status,
                    f"Worst slice {worst['slice']} water value = {format_num(worst['value'])} HU; "
                    f"reference {format_num(worst['reference'])} HU; "
                    f"tolerance [{format_num(worst['low'],2)},{format_num(worst['high'],2)}]. "
                    f"All slices: {slice_summary}",
                )
            )

        # -------- Homogeneity --------
        diff_rows = []
        current_slice = None
        current_low, current_high = None, None

        for line in lines:
            m_slice = re.match(r"^Slice\s+(\d+)$", line, re.I)
            if m_slice:
                current_slice = int(m_slice.group(1))
                continue

            low, high = parse_tolerance_line(line)
            if low is not None and high is not None:
                # Homogeneity acceptance is fixed ±4 HU, even if water value rows in same section use ±6.
                current_low, current_high = low, high
                continue

            m_inline = re.search(
                rf"^(Diff\.\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})$",
                line,
                re.I,
            )
            if m_inline:
                row = {
                    "slice": current_slice,
                    "position": m_inline.group(1),
                    "value": safe_float(m_inline.group(2)),
                    "reference": safe_float(m_inline.group(3)),
                    "low": safe_float(m_inline.group(4)),
                    "high": safe_float(m_inline.group(5)),
                }
                row["status"] = row_status(row["value"], row["low"], row["high"])
                diff_rows.append(row)
                continue

            m_simple = re.search(rf"^(Diff\.\d+)\s+({NUM})\s+({NUM})$", line, re.I)
            if m_simple and current_slice is not None:
                low = current_low if current_low is not None else -4.0
                high = current_high if current_high is not None else 4.0
                row = {
                    "slice": current_slice,
                    "position": m_simple.group(1),
                    "value": safe_float(m_simple.group(2)),
                    "reference": safe_float(m_simple.group(3)),
                    "low": low,
                    "high": high,
                }
                row["status"] = row_status(row["value"], row["low"], row["high"])
                diff_rows.append(row)

        if diff_rows:
            worst, overall_status = summarize_slice_rows(diff_rows, "Homogeneity", worst_by="max_abs")
            slice_summary = "; ".join(
                f"S{r['slice']} {r['position']}={format_num(r['value'])} HU [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
                for r in diff_rows
            )
            results.append(
                make_result(
                    "Homogeneity",
                    mode_name,
                    abs(worst["value"]),
                    "HU",
                    "Max peripheral-centre difference within ±4 HU",
                    overall_status,
                    f"Worst slice {worst['slice']} {worst['position']} = {format_num(worst['value'])} HU; "
                    f"tolerance [{format_num(worst['low'],2)},{format_num(worst['high'],2)}]. "
                    f"All differences: {slice_summary}",
                )
            )

    return results


def parse_ct_noise(text):
    section = extract_section(
        text,
        SECTION_PATTERNS["noise_start"],
        SECTION_PATTERNS["mtf_start"],
    )
    mode_blocks = split_mode_blocks(section, MODE_HEADER_PATTERNS["noise"])

    results = []
    for mode_name, block in mode_blocks:
        rows = []
        for line in iter_clean_lines(block):
            m = re.search(
                rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})(?:\s+(In Tol\.|Out Tol\.))?$",
                line,
                re.I,
            )
            if not m:
                continue

            row = {
                "slice": int(m.group(1)),
                "value": safe_float(m.group(2)),
                "reference": safe_float(m.group(3)),
                "low": safe_float(m.group(4)),
                "high": safe_float(m.group(5)),
            }
            row["status"] = row_status(row["value"], row["low"], row["high"])
            rows.append(row)

        if not rows:
            continue

        worst, overall_status = summarize_slice_rows(rows, "Noise", worst_by="max")
        slice_summary = "; ".join(
            f"S{r['slice']}={format_num(r['value'])} HU [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
            for r in rows
        )
        results.append(
            make_result(
                "Noise",
                mode_name,
                worst["value"],
                "HU",
                "Each slice must be within its slice-specific tolerance",
                overall_status,
                f"Worst slice {worst['slice']} noise = {format_num(worst['value'])} HU; "
                f"reference {format_num(worst['reference'])} HU; "
                f"tolerance [{format_num(worst['low'],2)},{format_num(worst['high'],2)}]. "
                f"All slices: {slice_summary}",
            )
        )

    return results


def parse_ct_mtf(text):
    section = extract_section(
        text,
        SECTION_PATTERNS["mtf_start"],
        SECTION_PATTERNS["table_start"],
    )
    mode_blocks = split_mode_blocks(section, MODE_HEADER_PATTERNS["mtf"])

    results = []
    for mode_name, block in mode_blocks:
        tol50 = (None, None)
        tol10 = (None, None)
        rows50 = []
        rows10 = []

        for line in iter_clean_lines(block):
            m_tol = re.search(
                rf"Tolerance:\s*({NUM})\s*{TOL_SEP}\s*({NUM})\s+Reference\s+Tolerance:\s*({NUM})\s*{TOL_SEP}\s*({NUM})",
                line,
                re.I,
            )
            if m_tol:
                tol50 = (safe_float(m_tol.group(1)), safe_float(m_tol.group(2)))
                tol10 = (safe_float(m_tol.group(3)), safe_float(m_tol.group(4)))
                continue

            if ("Tolerance" in line and "Reference Tolerance" in line) or ("Tolerance" in line and "Reference" in line and "..." in line):
                nums = [safe_float(x) for x in re.findall(NUM, line)]
                if len(nums) >= 4:
                    tol50 = (nums[0], nums[1])
                    tol10 = (nums[2], nums[3])
                    continue

            m_row = re.search(rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})$", line, re.I)
            if m_row:
                slice_no = int(m_row.group(1))
                val50 = safe_float(m_row.group(2))
                ref50 = safe_float(m_row.group(3))
                val10 = safe_float(m_row.group(4))
                ref10 = safe_float(m_row.group(5))

                row50 = {
                    "slice": slice_no,
                    "value": val50,
                    "reference": ref50,
                    "low": tol50[0],
                    "high": tol50[1],
                }
                row50["status"] = row_status(row50["value"], row50["low"], row50["high"])
                rows50.append(row50)

                row10 = {
                    "slice": slice_no,
                    "value": val10,
                    "reference": ref10,
                    "low": tol10[0],
                    "high": tol10[1],
                }
                row10["status"] = row_status(row10["value"], row10["low"], row10["high"])
                rows10.append(row10)
                continue

            # Sharpest mode fallback: slice value50 value10
            m_sharp = re.search(rf"^(\d+)\s+({NUM})\s+({NUM})$", line, re.I)
            if m_sharp and tol50[0] is not None and tol10[0] is not None:
                slice_no = int(m_sharp.group(1))
                val50 = safe_float(m_sharp.group(2))
                val10 = safe_float(m_sharp.group(3))

                row50 = {
                    "slice": slice_no,
                    "value": val50,
                    "reference": None,
                    "low": tol50[0],
                    "high": tol50[1],
                }
                row50["status"] = row_status(row50["value"], row50["low"], row50["high"])
                rows50.append(row50)

                row10 = {
                    "slice": slice_no,
                    "value": val10,
                    "reference": None,
                    "low": tol10[0],
                    "high": tol10[1],
                }
                row10["status"] = row_status(row10["value"], row10["low"], row10["high"])
                rows10.append(row10)

        if rows50:
            worst50, status50 = summarize_slice_rows(rows50, "MTF 50%", worst_by="min")
            slice_summary50 = "; ".join(
                f"S{r['slice']}={format_num(r['value'])} ref={format_num(r['reference'])} [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
                for r in rows50
            )
            results.append(
                make_result(
                    "MTF 50%",
                    mode_name,
                    worst50["value"],
                    "lp/cm",
                    "All slices within mode-specific MTF 50% tolerance",
                    status50,
                    f"Worst slice {worst50['slice']} MTF 50% = {format_num(worst50['value'])} lp/cm; "
                    f"reference {format_num(worst50['reference'])} lp/cm; "
                    f"tolerance [{format_num(worst50['low'],2)},{format_num(worst50['high'],2)}]. "
                    f"All slices: {slice_summary50}",
                )
            )

        if rows10:
            worst10, status10 = summarize_slice_rows(rows10, "MTF 10%", worst_by="min")
            slice_summary10 = "; ".join(
                f"S{r['slice']}={format_num(r['value'])} ref={format_num(r['reference'])} [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
                for r in rows10
            )
            results.append(
                make_result(
                    "MTF 10%",
                    mode_name,
                    worst10["value"],
                    "lp/cm",
                    "All slices within mode-specific MTF 10% tolerance",
                    status10,
                    f"Worst slice {worst10['slice']} MTF 10% = {format_num(worst10['value'])} lp/cm; "
                    f"reference {format_num(worst10['reference'])} lp/cm; "
                    f"tolerance [{format_num(worst10['low'],2)},{format_num(worst10['high'],2)}]. "
                    f"All slices: {slice_summary10}",
                )
            )

    return results


def parse_ct_table_positioning(text):
    section = extract_section(
        text,
        SECTION_PATTERNS["table_start"],
        SECTION_PATTERNS["tube_start"],
    )

    cont_rows = []
    step_rows = []

    for line in iter_clean_lines(section):
        m = re.search(
            rf"^Position\s+(\d+)\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})$",
            line,
            re.I,
        )
        if not m:
            continue

        pos = int(m.group(1))
        cont = safe_float(m.group(2))
        cont_low = safe_float(m.group(3))
        cont_high = safe_float(m.group(4))
        step = safe_float(m.group(5))
        step_low = safe_float(m.group(6))
        step_high = safe_float(m.group(7))

        rowc = {"position": pos, "value": cont, "low": cont_low, "high": cont_high}
        rowc["status"] = row_status(rowc["value"], rowc["low"], rowc["high"])
        cont_rows.append(rowc)

        rows = {"position": pos, "value": step, "low": step_low, "high": step_high}
        rows["status"] = row_status(rows["value"], rows["low"], rows["high"])
        step_rows.append(rows)

    results = []
    if cont_rows:
        worst = max(cont_rows, key=lambda x: abs(x["value"]))
        status = "PASS" if abs(worst["value"]) <= 1.0 and all(r["status"] == "PASS" for r in cont_rows) else "FAIL"
        pos_summary = "; ".join(
            f"P{r['position']}={format_num(r['value'])} mm [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
            for r in cont_rows
        )
        results.append(
            make_result(
                "Table Positioning (Continuous)",
                "Continuous movement",
                abs(worst["value"]),
                "mm",
                "Worst result within ±1 mm of expected position",
                status,
                f"Worst continuous result at position {worst['position']} = {format_num(worst['value'])} mm. "
                f"All positions: {pos_summary}",
            )
        )

    if step_rows:
        worst = max(step_rows, key=lambda x: abs(x["value"]))
        status = "PASS" if abs(worst["value"]) <= 1.0 and all(r["status"] == "PASS" for r in step_rows) else "FAIL"
        pos_summary = "; ".join(
            f"P{r['position']}={format_num(r['value'])} mm [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
            for r in step_rows
        )
        results.append(
            make_result(
                "Table Positioning (Stepwise)",
                "Stepwise movement",
                abs(worst["value"]),
                "mm",
                "Worst result within ±1 mm of expected position",
                status,
                f"Worst stepwise result at position {worst['position']} = {format_num(worst['value'])} mm. "
                f"All positions: {pos_summary}",
            )
        )

    return results


def parse_ct_tube_voltage(text):
    section = extract_section(
        text,
        SECTION_PATTERNS["tube_start"],
        SECTION_PATTERNS["image_start"],
    )

    rows = []
    for line in iter_clean_lines(section):
        m = re.search(
            rf"^(\d+)\s+(\d+)\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})(?:\s+(In Tol\.|Out Tol\.))?$",
            line,
            re.I,
        )
        if not m:
            continue

        nominal = safe_float(m.group(1))
        current = safe_float(m.group(2))
        measured = safe_float(m.group(3))
        low = safe_float(m.group(4))
        high = safe_float(m.group(5))

        row = {
            "nominal": nominal,
            "current": current,
            "measured": measured,
            "low": low,
            "high": high,
            "deviation": abs(measured - nominal),
        }
        row["status"] = row_status(row["measured"], row["low"], row["high"])
        rows.append(row)

    if not rows:
        return []

    worst = max(rows, key=lambda x: x["deviation"])
    overall_status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    row_summary = "; ".join(
        f"{int(r['nominal'])} kV -> {format_num(r['measured'])} kV [{format_num(r['low'],2)},{format_num(r['high'],2)}] {r['status']}"
        for r in rows
    )
    return [
        make_result(
            "Tube Voltage",
            "Minimum / middle / maximum tube voltages",
            worst["deviation"],
            "kV",
            "All measured kV within tolerance",
            overall_status,
            f"Worst deviation at nominal {int(worst['nominal'])} kV: measured {format_num(worst['measured'])} kV "
            f"(Δ={format_num(worst['deviation'])} kV). All rows: {row_summary}",
        )
    ]


def parse_ct_image_inspection(text):
    section = extract_section(text, SECTION_PATTERNS["image_start"], None)
    accept = len(re.findall(r"\bAccept\b", section, flags=re.I))
    reject = len(re.findall(r"\bReject\b|\bFail\b", section, flags=re.I))

    status = "PASS" if reject == 0 and accept > 0 else "FAIL"
    return [
        make_result(
            "Image Inspection",
            "Qualitative visual inspection",
            accept,
            "images",
            "All images must be accepted by the user",
            status,
            f"{accept} accepted, {reject} rejected. Qualitative test with no numeric tolerances.",
        )
    ]


def infer_ct_parsers_from_pdf_text(text):
    results = []
    results.extend(parse_ct_water_value_and_homogeneity(text))
    results.extend(parse_ct_noise(text))
    results.extend(parse_ct_mtf(text))
    results.extend(parse_ct_table_positioning(text))
    results.extend(parse_ct_tube_voltage(text))
    results.extend(parse_ct_image_inspection(text))
    return results


# =========================================================
# GITHUB HELPERS
# =========================================================
def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def github_get_file(owner, repo, path, token, branch="main"):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        resp = requests.get(url, headers=github_headers(token), params={"ref": branch}, timeout=30)
    except requests.RequestException as e:
        return None, None, f"GitHub connection error: {e}"

    if resp.status_code == 200:
        payload = resp.json()
        content = base64.b64decode(payload["content"]).decode("utf-8")
        return content, payload.get("sha"), None

    if resp.status_code == 404:
        return None, None, None

    return None, None, f"GitHub read error {resp.status_code}: {resp.text}"


def github_put_file(owner, repo, path, token, content_text, message, branch="main", sha=None):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        payload = {
            "message": message,
            "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=github_headers(token), json=payload, timeout=30)
    except requests.RequestException as e:
        return False, f"GitHub connection error: {e}"

    if resp.status_code in (200, 201):
        return True, None

    return False, f"GitHub write error {resp.status_code}: {resp.text}"


def github_delete_file(owner, repo, path, token, message, branch="main", sha=None):
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        payload = {
            "message": message,
            "branch": branch,
            "sha": sha,
        }
        resp = requests.delete(url, headers=github_headers(token), json=payload, timeout=30)
    except requests.RequestException as e:
        return False, f"GitHub connection error: {e}"

    if resp.status_code in (200, 204):
        return True, None

    return False, f"GitHub delete error {resp.status_code}: {resp.text}"


def load_history_from_github(owner, repo, path, token, branch="main"):
    content, sha, err = github_get_file(owner, repo, path, token, branch=branch)
    if err:
        return empty_history_df(), None, err

    if content is None or not content.strip():
        return empty_history_df(), None, None

    try:
        df = pd.read_csv(io.StringIO(content))
    except pd.errors.EmptyDataError:
        return empty_history_df(), None, None
    except Exception as e:
        return empty_history_df(), None, f"Could not parse GitHub CSV: {e}"

    return normalize_history_df(df), sha, None


def save_history_to_github(df, owner, repo, path, token, branch="main", sha=None):
    csv_text = normalize_history_df(df).to_csv(index=False)
    ok, err = github_put_file(
        owner=owner,
        repo=repo,
        path=path,
        token=token,
        content_text=csv_text,
        message="Update CT QC history",
        branch=branch,
        sha=sha,
    )
    return ok, err


# =========================================================
# LOCKING
# =========================================================
def acquire_local_lock(lock_path=LOCAL_LOCK_FILE, timeout_seconds=20, stale_lock_seconds=300):
    start = time.time()
    while True:
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_lock_seconds:
                try:
                    lock_path.unlink()
                except Exception:
                    pass

        try:
            with open(lock_path, "x", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "lock_id": str(uuid.uuid4()),
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                )
            return True
        except FileExistsError:
            pass
        except Exception:
            pass

        if time.time() - start > timeout_seconds:
            return False

        time.sleep(0.5)


def release_local_lock(lock_path=LOCAL_LOCK_FILE):
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def github_lock_path(csv_path):
    csv_path = csv_path.strip("/")
    if "/" in csv_path:
        parent = csv_path.rsplit("/", 1)[0]
        return f"{parent}/ct_qc_history.lock.json"
    return "ct_qc_history.lock.json"


def acquire_github_lock(owner, repo, csv_path, token, branch="main", timeout_seconds=25, stale_lock_seconds=300):
    lock_path = github_lock_path(csv_path)
    start = time.time()

    while True:
        content, sha, err = github_get_file(owner, repo, lock_path, token, branch=branch)
        now = datetime.now()

        if err:
            return False, None, f"GitHub lock read error: {err}"

        if content is not None:
            try:
                payload = json.loads(content)
                created_at = datetime.fromisoformat(payload.get("created_at"))
                if (now - created_at).total_seconds() > stale_lock_seconds:
                    github_delete_file(
                        owner=owner,
                        repo=repo,
                        path=lock_path,
                        token=token,
                        message="Remove stale CT QC lock",
                        branch=branch,
                        sha=sha,
                    )
                else:
                    if time.time() - start > timeout_seconds:
                        return False, None, "Could not acquire GitHub lock. Another user may be saving right now."
                    time.sleep(1.0)
                    continue
            except Exception:
                if time.time() - start > timeout_seconds:
                    return False, None, "Could not interpret existing GitHub lock file."
                time.sleep(1.0)
                continue

        lock_payload = json.dumps(
            {
                "lock_id": str(uuid.uuid4()),
                "created_at": now.isoformat(timespec="seconds"),
                "owner": owner,
                "repo": repo,
                "path": csv_path,
            },
            indent=2,
        )

        ok, _ = github_put_file(
            owner=owner,
            repo=repo,
            path=lock_path,
            token=token,
            content_text=lock_payload,
            message="Acquire CT QC lock",
            branch=branch,
            sha=None,
        )

        if ok:
            _, latest_sha, latest_err = github_get_file(owner, repo, lock_path, token, branch=branch)
            if latest_err:
                return False, None, latest_err
            return True, latest_sha, None

        if time.time() - start > timeout_seconds:
            return False, None, "Could not acquire GitHub lock. Another user may be saving right now."

        time.sleep(1.0)


def release_github_lock(owner, repo, csv_path, token, branch="main"):
    lock_path = github_lock_path(csv_path)
    content, sha, err = github_get_file(owner, repo, lock_path, token, branch=branch)
    if err:
        return False, err
    if content is None or not sha:
        return True, None

    return github_delete_file(
        owner=owner,
        repo=repo,
        path=lock_path,
        token=token,
        message="Release CT QC lock",
        branch=branch,
        sha=sha,
    )


# =========================================================
# HISTORY STORAGE
# =========================================================
def load_history(local_only=True, github_cfg=None):
    if not local_only and github_cfg:
        return load_history_from_github(
            github_cfg["owner"],
            github_cfg["repo"],
            github_cfg["path"],
            github_cfg["token"],
            branch=github_cfg["branch"],
        )

    if LOCAL_HISTORY_CSV.exists():
        try:
            if LOCAL_HISTORY_CSV.stat().st_size == 0:
                df = empty_history_df()
            else:
                df = pd.read_csv(LOCAL_HISTORY_CSV)
        except pd.errors.EmptyDataError:
            df = empty_history_df()
        except Exception as e:
            return empty_history_df(), None, f"Could not read local history CSV: {e}"
    else:
        df = empty_history_df()

    return normalize_history_df(df), None, None


@st.cache_data(ttl=60, show_spinner=False)
def cached_load_history(local_only=True, github_cfg=None):
    return load_history(local_only=local_only, github_cfg=github_cfg)


def save_history_local(df):
    normalize_history_df(df).to_csv(LOCAL_HISTORY_CSV, index=False)


def append_results_to_history(
    results,
    session_label,
    timestamp,
    site_name,
    scanner_name,
    scanner_id,
    local_only=True,
    github_cfg=None,
    sha=None,
):
    history, _, _ = load_history(local_only=local_only, github_cfg=github_cfg)

    rows = []
    for r in results:
        rows.append(
            {
                "timestamp": timestamp,
                "session_label": session_label,
                "site_name": site_name,
                "scanner_name": scanner_name,
                "scanner_id": scanner_id,
                "test_name": r["test_name"],
                "value": r["value"],
                "unit": r["unit"],
                "criteria": r["criteria"],
                "status": r["status"],
                "details": r["details"],
                "source_file": r.get("source_file", ""),
                "sequence_label": r.get("sequence_label", ""),
            }
        )

    updated = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
    updated = normalize_history_df(updated)
    updated = updated.drop_duplicates(
        subset=["scanner_id", "timestamp", "test_name", "sequence_label", "source_file"],
        keep="last",
    )

    if local_only:
        save_history_local(updated)
        return updated, None, None

    ok, err = save_history_to_github(
        updated,
        github_cfg["owner"],
        github_cfg["repo"],
        github_cfg["path"],
        github_cfg["token"],
        branch=github_cfg["branch"],
        sha=sha,
    )
    return updated, ok, err


def save_results_with_lock(
    results,
    session_label,
    timestamp,
    site_name,
    scanner_name,
    scanner_id,
    local_only=True,
    github_cfg=None,
):
    if local_only:
        locked = acquire_local_lock()
        if not locked:
            return None, "Could not acquire local file lock. Please try again."
        try:
            updated, _, _ = append_results_to_history(
                results,
                session_label,
                timestamp,
                site_name,
                scanner_name,
                scanner_id,
                local_only=True,
                github_cfg=None,
                sha=None,
            )
            return updated, None
        finally:
            release_local_lock()

    ok, _, lock_err = acquire_github_lock(
        github_cfg["owner"],
        github_cfg["repo"],
        github_cfg["path"],
        github_cfg["token"],
        branch=github_cfg["branch"],
    )
    if not ok:
        return None, lock_err

    try:
        _, existing_sha, load_err = load_history(local_only=False, github_cfg=github_cfg)
        if load_err:
            return None, load_err

        updated, _, save_err = append_results_to_history(
            results,
            session_label,
            timestamp,
            site_name,
            scanner_name,
            scanner_id,
            local_only=False,
            github_cfg=github_cfg,
            sha=existing_sha,
        )
        if save_err:
            return None, save_err

        return updated, None
    finally:
        release_github_lock(
            github_cfg["owner"],
            github_cfg["repo"],
            github_cfg["path"],
            github_cfg["token"],
            branch=github_cfg["branch"],
        )


# =========================================================
# TREND DATA PREP
# =========================================================
def build_frontpage_trend_df(history_df, include_current_df=None):
    trend_df = history_df.copy()

    if include_current_df is not None and not include_current_df.empty:
        trend_df = pd.concat([trend_df, include_current_df], ignore_index=True)

    trend_df = normalize_history_df(trend_df)
    if trend_df.empty:
        return trend_df

    trend_df["timestamp_dt"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
    trend_df = trend_df.dropna(subset=["timestamp_dt"])

    out = trend_df.sort_values(["scanner_id", "test_name", "sequence_label", "timestamp_dt"]).reset_index(drop=True)
    out["trend_label"] = out.apply(
        lambda r: f"{r['test_name']} | {r['sequence_label']}" if str(r["sequence_label"]).strip() else r["test_name"],
        axis=1,
    )
    return out


# =========================================================
# PDF STYLES / HELPERS
# =========================================================
def get_pdf_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183A63"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubTitleCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=8,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeadingCustom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#183A63"),
            spaceBefore=6,
            spaceAfter=6,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetaCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            alignment=TA_LEFT,
            textColor=colors.black,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCellCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.black,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeaderCustom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.white,
            wordWrap="LTR",
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PassBadge",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#166534"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="FailBadge",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#991B1B"),
        )
    )

    return styles


def add_pdf_header(elements, styles, title, subtitle="", site_name="", scanner_name="", include_logo=True):
    if include_logo and LOGO_PATH.exists():
        try:
            logo = RLImage(str(LOGO_PATH), width=110, height=40)
            elements.append(logo)
            elements.append(Spacer(1, 6))
        except Exception:
            pass

    elements.append(Paragraph(title, styles["ReportTitleCustom"]))
    if subtitle:
        elements.append(Paragraph(subtitle, styles["ReportSubTitleCustom"]))
    if site_name or scanner_name:
        meta_line = " | ".join([x for x in [site_name, scanner_name] if x])
        if meta_line:
            elements.append(Paragraph(meta_line, styles["ReportSubTitleCustom"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
    elements.append(Spacer(1, 8))


def status_paragraph(status, styles):
    s = str(status).upper().strip()
    if s == "PASS":
        return Paragraph("PASS", styles["PassBadge"])
    return Paragraph("FAIL", styles["FailBadge"])


def format_value_unit(value, unit):
    if pd.isna(value):
        return ""
    return f"{format_num(value)} {unit}".strip()


def format_session_date(ts):
    ts = str(ts)
    return ts.split("T")[0] if ts else ""


def display_test_label(row):
    seq = str(row.get("sequence_label", "")).strip()
    test = str(row.get("test_name", "")).strip()
    return f"{test} | {seq}" if seq else test


def build_results_table(results_df, styles):
    df = normalize_history_df(results_df).copy()
    df = sort_tests_ct(df)

    cell_style = styles["TableCellCustom"]
    header_style = styles["TableHeaderCustom"]

    table_data = [[
        Paragraph("Test / Mode", header_style),
        Paragraph("Value", header_style),
        Paragraph("Tolerance / Criteria", header_style),
        Paragraph("Status", header_style),
    ]]

    for _, row in df.iterrows():
        value_text = format_value_unit(row["value"], row["unit"])
        criteria_text = str(row["criteria"]) if pd.notna(row["criteria"]) else ""
        test_text = display_test_label(row)

        table_data.append([
            Paragraph(test_text, cell_style),
            Paragraph(value_text, cell_style),
            Paragraph(criteria_text, cell_style),
            status_paragraph(row["status"], styles),
        ])

    table = Table(
        table_data,
        colWidths=[220, 80, 160, 50],
        repeatRows=1,
        splitByRow=1,
    )

    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#EEF3F8")]),
    ])

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if str(row["status"]).upper() == "PASS":
            ts.add("BACKGROUND", (3, idx), (3, idx), colors.HexColor("#DCFCE7"))
        else:
            ts.add("BACKGROUND", (3, idx), (3, idx), colors.HexColor("#FEE2E2"))

    table.setStyle(ts)
    return table


def fig_to_rl_image(fig, width=500):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    img_reader = ImageReader(buf)
    iw, ih = img_reader.getSize()
    aspect = ih / float(iw) if iw else 0.58
    return RLImage(buf, width=width, height=width * aspect)


def add_reference_lines_ct(ax, trend_label):
    if trend_label.startswith("Water Value"):
        ax.axhline(6.0, linestyle="--", alpha=0.7)
        ax.axhline(-6.0, linestyle="--", alpha=0.7)
    elif trend_label.startswith("Homogeneity"):
        ax.axhline(4.0, linestyle="--", alpha=0.7)
        ax.axhline(-4.0, linestyle="--", alpha=0.7)
    elif trend_label.startswith("Table Positioning"):
        ax.axhline(1.0, linestyle="--", alpha=0.7)


def create_trend_chart(df, trend_label):
    sub = build_frontpage_trend_df(df)
    if sub.empty:
        return None

    sub = sub[sub["trend_label"] == trend_label].copy()
    sub = sub.dropna(subset=["timestamp_dt", "value"]).sort_values("timestamp_dt")
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(sub["timestamp_dt"], sub["value"], marker="o")
    ax.set_title(trend_label)
    unit = sub["unit"].dropna().iloc[0] if not sub["unit"].dropna().empty else ""
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(f"Value ({unit})")
    ax.grid(True, alpha=0.3)
    add_reference_lines_ct(ax, trend_label)
    fig.autofmt_xdate()
    return fig


def build_pdf_report(
    results_df,
    history_df,
    site_name,
    scanner_name,
    session_label,
    timestamp_str,
):
    safe_scanner = sanitize_filename(scanner_name or "scanner")
    safe_date = format_session_date(timestamp_str) or datetime.now().strftime("%Y-%m-%d")
    pdf_path = REPORTS_DIR / f"CT_QC_Report_{safe_scanner}_{safe_date}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )
    styles = get_pdf_styles()
    elements = []

    results_df = normalize_history_df(results_df).copy()
    results_df = sort_tests_ct(results_df)

    add_pdf_header(
        elements,
        styles,
        title="CT IEC Constancy QC Compliance Report",
        subtitle="Formal session summary with parsed measurements and trend review",
        site_name=site_name,
        scanner_name=scanner_name,
        include_logo=True,
    )

    elements.append(Paragraph("Session Information", styles["SectionHeadingCustom"]))
    elements.append(Paragraph(f"<b>Session label:</b> {session_label}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Timestamp:</b> {timestamp_str}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Session date:</b> {format_session_date(timestamp_str)}", styles["MetaCustom"]))
    elements.append(Spacer(1, 8))

    overall = "PASS" if (results_df["status"] == "PASS").all() else "FAIL"
    overall_color = "#166534" if overall == "PASS" else "#991B1B"
    elements.append(
        Paragraph(
            f'<font color="{overall_color}"><b>Overall result: {overall}</b></font>',
            styles["SectionHeadingCustom"],
        )
    )
    elements.append(Spacer(1, 4))

    elements.append(Paragraph("Session Results Summary", styles["SectionHeadingCustom"]))
    elements.append(build_results_table(results_df, styles))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Parsed Details", styles["SectionHeadingCustom"]))
    for _, row in results_df.iterrows():
        label = display_test_label(row)
        elements.append(Paragraph(f"<b>{label}:</b> {row['details']}", styles["MetaCustom"]))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Trend Charts", styles["SectionHeadingCustom"]))
    added_any_chart = False

    trend_labels = []
    for _, row in results_df.iterrows():
        lbl = display_test_label(row)
        if lbl not in trend_labels:
            trend_labels.append(lbl)

    hist = build_frontpage_trend_df(history_df)
    for trend_label in trend_labels:
        fig = create_trend_chart(hist, trend_label)
        if fig is not None:
            added_any_chart = True
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(trend_label, styles["MetaCustom"]))
            elements.append(fig_to_rl_image(fig, width=500))
            plt.close(fig)
            elements.append(Spacer(1, 8))

    if not added_any_chart:
        elements.append(Paragraph("No historical numeric data yet for trend charts.", styles["MetaCustom"]))

    doc.build(elements)
    return pdf_path


def build_session_summary_pdf(history_df, site_name=None, scanner_name=None, scanner_id=None):
    scanner_fragment = sanitize_filename(scanner_name or scanner_id or "scanner")
    pdf_path = REPORTS_DIR / f"CT_QC_Session_Summary_{scanner_fragment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )

    styles = get_pdf_styles()
    elements = []

    df = normalize_history_df(history_df).copy()
    if df.empty:
        add_pdf_header(
            elements,
            styles,
            title="CT IEC Constancy QC Session Summary",
            subtitle="Historical report",
            site_name=site_name or "",
            scanner_name=scanner_name or "",
            include_logo=True,
        )
        elements.append(Paragraph("No history data available.", styles["MetaCustom"]))
        doc.build(elements)
        return pdf_path

    if scanner_id:
        df = df[df["scanner_id"] == scanner_id].copy()
    else:
        if site_name:
            df = df[df["site_name"] == site_name].copy()
        if scanner_name:
            df = df[df["scanner_name"] == scanner_name].copy()

    if df.empty:
        add_pdf_header(
            elements,
            styles,
            title="CT IEC Constancy QC Session Summary",
            subtitle="Historical report",
            site_name=site_name or "",
            scanner_name=scanner_name or "",
            include_logo=True,
        )
        elements.append(Paragraph("No matching session history found for the selected scanner.", styles["MetaCustom"]))
        doc.build(elements)
        return pdf_path

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_dt", ascending=False)

    add_pdf_header(
        elements,
        styles,
        title="CT IEC Constancy QC Session Summary",
        subtitle="All recorded sessions for the selected system",
        site_name=site_name or "",
        scanner_name=scanner_name or "",
        include_logo=True,
    )

    group_cols = ["timestamp", "session_label", "site_name", "scanner_name", "scanner_id"]
    grouped_items = list(df.groupby(group_cols, sort=False))

    for idx, ((timestamp, session_label, g_site, g_scanner, g_scanner_id), g) in enumerate(grouped_items):
        g = sort_tests_ct(g.copy())
        overall = "PASS" if (g["status"] == "PASS").all() else "FAIL"
        overall_color = "#166534" if overall == "PASS" else "#991B1B"

        elements.append(Paragraph(f"Session {idx + 1}", styles["SectionHeadingCustom"]))
        elements.append(Paragraph(f"<b>Session date:</b> {format_session_date(timestamp)}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Timestamp:</b> {timestamp}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Session label:</b> {session_label}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Site:</b> {g_site}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>Scanner:</b> {g_scanner}", styles["MetaCustom"]))
        elements.append(Paragraph(f"<b>System ID:</b> {g_scanner_id}", styles["MetaCustom"]))
        elements.append(
            Paragraph(
                f'<font color="{overall_color}"><b>Overall result:</b> {overall}</font>',
                styles["MetaCustom"],
            )
        )
        elements.append(Spacer(1, 8))
        elements.append(build_results_table(g, styles))

        if idx < len(grouped_items) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    return pdf_path


def build_single_session_pdf(session_df):
    pdf_path = REPORTS_DIR / f"CT_QC_Selected_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30,
    )

    styles = get_pdf_styles()
    elements = []

    df = normalize_history_df(session_df).copy()
    df = sort_tests_ct(df)

    if df.empty:
        add_pdf_header(
            elements,
            styles,
            title="CT IEC Constancy QC Session Report",
            subtitle="Selected historical session",
            include_logo=True,
        )
        elements.append(Paragraph("No data found for the selected session.", styles["MetaCustom"]))
        doc.build(elements)
        return pdf_path

    first = df.iloc[0]
    overall = "PASS" if (df["status"] == "PASS").all() else "FAIL"
    overall_color = "#166534" if overall == "PASS" else "#991B1B"

    add_pdf_header(
        elements,
        styles,
        title="CT IEC Constancy QC Session Report",
        subtitle="Formal single-session report generated from stored history",
        site_name=str(first["site_name"]),
        scanner_name=str(first["scanner_name"]),
        include_logo=True,
    )

    elements.append(Paragraph("Session Information", styles["SectionHeadingCustom"]))
    elements.append(Paragraph(f"<b>Session date:</b> {format_session_date(first['timestamp'])}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Timestamp:</b> {first['timestamp']}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>Session label:</b> {first['session_label']}", styles["MetaCustom"]))
    elements.append(Paragraph(f"<b>System ID:</b> {first['scanner_id']}", styles["MetaCustom"]))
    elements.append(Spacer(1, 8))

    elements.append(
        Paragraph(
            f'<font color="{overall_color}"><b>Overall result: {overall}</b></font>',
            styles["SectionHeadingCustom"],
        )
    )
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Results Summary", styles["SectionHeadingCustom"]))
    elements.append(build_results_table(df, styles))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Details", styles["SectionHeadingCustom"]))
    for _, row in df.iterrows():
        elements.append(Paragraph(f"<b>{display_test_label(row)}:</b> {row['details']}", styles["MetaCustom"]))
        elements.append(Spacer(1, 4))

    doc.build(elements)
    return pdf_path


# =========================================================
# APP
# =========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Upload one Siemens IEC Constancy PDF, parse CT IEC Constancy tests, "
    "save history with timestamp, and generate PDF reports from current or historical sessions."
)

if "session_saved" not in st.session_state:
    st.session_state.session_saved = False

if "parsed_results" not in st.session_state:
    st.session_state.parsed_results = []

if "combined_results" not in st.session_state:
    st.session_state.combined_results = []

if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None

if "pdf_report_bytes" not in st.session_state:
    st.session_state.pdf_report_bytes = None
    st.session_state.pdf_report_name = None

if "summary_pdf_bytes" not in st.session_state:
    st.session_state.summary_pdf_bytes = None
    st.session_state.summary_pdf_name = None

if "selected_session_pdf_bytes" not in st.session_state:
    st.session_state.selected_session_pdf_bytes = None
    st.session_state.selected_session_pdf_name = None

try:
    SECRET_GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except Exception:
    SECRET_GITHUB_TOKEN = ""

try:
    repo_full = st.secrets["GITHUB_REPO"]
    owner, repo_name = repo_full.split("/", 1)
except Exception:
    owner = DEFAULT_GITHUB_OWNER
    repo_name = DEFAULT_GITHUB_REPO

try:
    github_branch = st.secrets["GITHUB_BRANCH"]
except Exception:
    github_branch = DEFAULT_GITHUB_BRANCH

github_cfg = {
    "owner": owner,
    "repo": repo_name,
    "branch": github_branch,
    "path": DEFAULT_GITHUB_CSV_PATH,
    "token": SECRET_GITHUB_TOKEN,
}
USE_GITHUB = github_is_ready(github_cfg)

history_df, _, preload_err = cached_load_history(
    local_only=not USE_GITHUB,
    github_cfg=github_cfg if USE_GITHUB else None,
)
if preload_err:
    st.error(preload_err)
    history_df = empty_history_df()

history_df = normalize_history_df(history_df)
known_sites = sorted([x for x in history_df["site_name"].unique().tolist() if x])

uploaded_file = st.file_uploader(
    "Upload Siemens IEC Constancy PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

pdf_text = ""
pdf_meta = {
    "site_name": "",
    "scanner_name": "",
    "serial_number": "",
    "timestamp_iso": "",
}

uploaded_signature = ""
if uploaded_file:
    try:
        pdf_text = read_pdf_text(uploaded_file)
        pdf_text = normalize_pdf_text(pdf_text)
        pdf_meta = extract_pdf_metadata(pdf_text)

        if DEBUG_MODE:
            with st.expander("DEBUG: extracted PDF text"):
                st.text(pdf_text[:16000])

            debug_dump_sections(pdf_text)

        uploaded_file.seek(0)
        uploaded_signature = str(hash(uploaded_file.getvalue()))
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()

with st.sidebar:
    st.header("Session info")

    suggested_site = pdf_meta.get("site_name", "") if uploaded_file else ""
    suggested_scanner = pdf_meta.get("scanner_name", "") if uploaded_file else ""
    suggested_timestamp = pdf_meta.get("timestamp_iso", "") if uploaded_file else ""

    if known_sites:
        site_mode = st.radio("Site entry mode", ["Select existing", "Enter new"], horizontal=False)
        if site_mode == "Select existing":
            default_site_index = 0
            if suggested_site and suggested_site in known_sites:
                default_site_index = known_sites.index(suggested_site)
            site_name = st.selectbox("Site / Hospital", options=known_sites, index=default_site_index)
        else:
            site_name = st.text_input("Site / Hospital", value=suggested_site)
    else:
        site_name = st.text_input("Site / Hospital", value=suggested_site)

    filtered_scanners = []
    if site_name.strip():
        filtered_scanners = sorted(
            [
                x
                for x in history_df.loc[history_df["site_name"] == site_name.strip(), "scanner_name"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if x
            ]
        )

    if suggested_scanner and suggested_scanner not in filtered_scanners and suggested_scanner.strip():
        filtered_scanners = sorted(filtered_scanners + [suggested_scanner])

    if filtered_scanners:
        scanner_mode = st.radio("Scanner entry mode", ["Select existing", "Enter new"], horizontal=False)
        if scanner_mode == "Select existing":
            default_scanner_index = 0
            if suggested_scanner and suggested_scanner in filtered_scanners:
                default_scanner_index = filtered_scanners.index(suggested_scanner)
            scanner_name = st.selectbox("Scanner / System", options=filtered_scanners, index=default_scanner_index)
        else:
            scanner_name = st.text_input("Scanner / System", value=suggested_scanner)
    else:
        scanner_name = st.text_input("Scanner / System", value=suggested_scanner)

    scanner_id = build_scanner_id(site_name, scanner_name)
    st.caption(f"System ID: {scanner_id}")

    default_label = "IEC Constancy QC"
    session_label = st.text_input("Session label", value=default_label)

    timestamp_default = suggested_timestamp or datetime.now().isoformat(timespec="seconds")
    custom_timestamp = st.text_input("Timestamp (optional, ISO format)", value=timestamp_default)

    if custom_timestamp.strip():
        if not validate_iso_timestamp(custom_timestamp.strip()):
            st.error("Timestamp must be valid ISO format, e.g. 2026-04-02T14:16:35")
            st.stop()
        timestamp_str = custom_timestamp.strip()
    else:
        timestamp_str = datetime.now().isoformat(timespec="seconds")

    if uploaded_file and pdf_meta.get("serial_number"):
        st.caption(f"PDF Serial Number: {pdf_meta['serial_number']}")

    if USE_GITHUB:
        st.success("GitHub history storage is active.")
    else:
        st.warning("GitHub not fully configured. Using local history file.")

current_upload_signature = (
    uploaded_signature,
    site_name.strip(),
    scanner_name.strip(),
    session_label.strip(),
    timestamp_str.strip(),
)

if st.session_state.last_upload_signature != current_upload_signature:
    st.session_state.session_saved = False
    st.session_state.last_upload_signature = current_upload_signature

parsed_results = []
results_df = pd.DataFrame()

if uploaded_file:
    with st.spinner("Parsing IEC Constancy PDF..."):
        if DEBUG_MODE:
            with st.expander("DEBUG: section checks"):
                st.write(
                    "Homogeneity section found:",
                    bool(extract_section(pdf_text, SECTION_PATTERNS["homogeneity_start"], SECTION_PATTERNS["noise_start"])),
                )
                st.write(
                    "Noise section found:",
                    bool(extract_section(pdf_text, SECTION_PATTERNS["noise_start"], SECTION_PATTERNS["mtf_start"])),
                )
                st.write(
                    "MTF section found:",
                    bool(extract_section(pdf_text, SECTION_PATTERNS["mtf_start"], SECTION_PATTERNS["table_start"])),
                )
                st.write(
                    "Table Positioning section found:",
                    bool(extract_section(pdf_text, SECTION_PATTERNS["table_start"], SECTION_PATTERNS["tube_start"])),
                )
                st.write(
                    "Tube Voltage section found:",
                    bool(extract_section(pdf_text, SECTION_PATTERNS["tube_start"], SECTION_PATTERNS["image_start"])),
                )
                st.write(
                    "Image Inspection section found:",
                    bool(extract_section(pdf_text, SECTION_PATTERNS["image_start"], None)),
                )

        parsed_results = infer_ct_parsers_from_pdf_text(pdf_text)

    for r in parsed_results:
        r["source_file"] = uploaded_file.name

    st.session_state.parsed_results = parsed_results
    st.session_state.combined_results = parsed_results

    results_df = pd.DataFrame(parsed_results)
    results_df = sort_tests_ct(results_df)

    st.subheader("Current CT session results")
    display_cols = [c for c in ["test_name", "sequence_label", "value", "unit", "criteria", "status", "details", "source_file"] if c in results_df.columns]
    st.dataframe(results_df[display_cols], width="stretch")

    overall = "PASS" if (results_df["status"] == "PASS").all() else "FAIL"
    st.metric("Overall session result", overall)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save session to history", type="primary", key="save_session_to_history"):
            if not site_name.strip() or not scanner_name.strip():
                st.error("Please enter both Site / Hospital and Scanner / System.")
            else:
                history_after_save, save_err = save_results_with_lock(
                    parsed_results,
                    session_label,
                    timestamp_str,
                    site_name,
                    scanner_name,
                    scanner_id,
                    local_only=not USE_GITHUB,
                    github_cfg=github_cfg if USE_GITHUB else None,
                )
                if save_err:
                    st.error(save_err)
                else:
                    st.session_state.session_saved = True
                    history_df = normalize_history_df(history_after_save)
                    cached_load_history.clear()
                    st.success(f"Saved {len(parsed_results)} CT QC results to history for system: {scanner_id}")

    with col2:
        if st.button("Generate PDF report", key="generate_pdf_report"):
            if not site_name.strip() or not scanner_name.strip():
                st.error("Please enter both Site / Hospital and Scanner / System.")
            else:
                temp_history = history_df.copy()
                if uploaded_file and parsed_results:
                    current_rows = pd.DataFrame(
                        [
                            {
                                "timestamp": timestamp_str,
                                "session_label": session_label,
                                "site_name": site_name,
                                "scanner_name": scanner_name,
                                "scanner_id": scanner_id,
                                "test_name": r["test_name"],
                                "value": r["value"],
                                "unit": r["unit"],
                                "criteria": r["criteria"],
                                "status": r["status"],
                                "details": r["details"],
                                "source_file": r.get("source_file", ""),
                                "sequence_label": r.get("sequence_label", ""),
                            }
                            for r in parsed_results
                        ]
                    )
                    temp_history = pd.concat([temp_history, current_rows], ignore_index=True)
                    temp_history = normalize_history_df(temp_history)

                pdf_path = build_pdf_report(
                    results_df=results_df,
                    history_df=temp_history,
                    site_name=site_name,
                    scanner_name=scanner_name,
                    session_label=session_label,
                    timestamp_str=timestamp_str,
                )

                with open(pdf_path, "rb") as f:
                    st.session_state.pdf_report_bytes = f.read()
                    st.session_state.pdf_report_name = pdf_path.name

                st.success(f"PDF report created: {pdf_path.name}")

    with col3:
        if st.button("Generate session summary PDF", key="generate_session_summary_pdf"):
            if not site_name.strip() or not scanner_name.strip():
                st.error("Please enter both Site / Hospital and Scanner / System.")
            else:
                summary_history = history_df.copy()

                if uploaded_file and parsed_results and not st.session_state.session_saved:
                    current_rows = pd.DataFrame(
                        [
                            {
                                "timestamp": timestamp_str,
                                "session_label": session_label,
                                "site_name": site_name,
                                "scanner_name": scanner_name,
                                "scanner_id": scanner_id,
                                "test_name": r["test_name"],
                                "value": r["value"],
                                "unit": r["unit"],
                                "criteria": r["criteria"],
                                "status": r["status"],
                                "details": r["details"],
                                "source_file": r.get("source_file", ""),
                                "sequence_label": r.get("sequence_label", ""),
                            }
                            for r in parsed_results
                        ]
                    )
                    summary_history = pd.concat([summary_history, current_rows], ignore_index=True)
                    summary_history = normalize_history_df(summary_history)

                pdf_path = build_session_summary_pdf(
                    summary_history,
                    site_name=site_name,
                    scanner_name=scanner_name,
                    scanner_id=scanner_id,
                )

                with open(pdf_path, "rb") as f:
                    st.session_state.summary_pdf_bytes = f.read()
                    st.session_state.summary_pdf_name = pdf_path.name

                st.success(f"Session summary PDF created: {pdf_path.name}")

    if st.session_state.pdf_report_bytes:
        st.download_button(
            "Download PDF report",
            data=st.session_state.pdf_report_bytes,
            file_name=st.session_state.pdf_report_name,
            mime="application/pdf",
            key="download_pdf_report",
        )

    if st.session_state.summary_pdf_bytes:
        st.download_button(
            "Download session summary PDF",
            data=st.session_state.summary_pdf_bytes,
            file_name=st.session_state.summary_pdf_name,
            mime="application/pdf",
            key="download_summary_pdf",
        )

else:
    parsed_results = st.session_state.get("parsed_results", [])
    if parsed_results:
        results_df = pd.DataFrame(parsed_results)
        results_df = sort_tests_ct(results_df)

# =========================================================
# TREND DATA PREP
# =========================================================
history_df, _, load_err = cached_load_history(
    local_only=not USE_GITHUB,
    github_cfg=github_cfg if USE_GITHUB else None,
)
if load_err:
    st.error(load_err)
    history_df = empty_history_df()

history_df = normalize_history_df(history_df)

current_rows_df = pd.DataFrame()
if uploaded_file and parsed_results and not st.session_state.session_saved:
    current_rows_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp_str,
                "session_label": session_label,
                "site_name": site_name,
                "scanner_name": scanner_name,
                "scanner_id": scanner_id,
                "test_name": r["test_name"],
                "value": r["value"],
                "unit": r["unit"],
                "criteria": r["criteria"],
                "status": r["status"],
                "details": r["details"],
                "source_file": r.get("source_file", ""),
                "sequence_label": r.get("sequence_label", ""),
            }
            for r in parsed_results
        ]
    )

front_trend_df = build_frontpage_trend_df(history_df, include_current_df=current_rows_df)

# =========================================================
# FRONT PAGE SINGLE TREND PANEL
# =========================================================
st.subheader("Trend preview")

if front_trend_df.empty:
    st.info("No trend data available yet.")
else:
    panel_col1, panel_col2 = st.columns(2)

    system_options = sorted(front_trend_df["scanner_id"].dropna().astype(str).unique().tolist())

    with panel_col1:
        default_idx = 0
        if "scanner_id" in locals() and scanner_id in system_options:
            default_idx = system_options.index(scanner_id)

        selected_system = st.selectbox(
            "Select system",
            system_options,
            index=default_idx,
            key="front_system_select",
        )

    system_df = front_trend_df[front_trend_df["scanner_id"] == selected_system].copy()
    system_df = sort_tests_ct(system_df)

    trend_options = system_df["trend_label"].dropna().astype(str).unique().tolist()
    trend_options = sorted(trend_options)

    with panel_col2:
        selected_trend = st.selectbox(
            "Select test / mode",
            trend_options,
            key="front_trend_select",
        )

    timestamp_options = (
        front_trend_df.loc[front_trend_df["scanner_id"] == selected_system, "timestamp"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    timestamp_options = sorted(timestamp_options, reverse=True)

    if timestamp_options:
        selected_timestamp = st.selectbox(
            "Select session timestamp",
            timestamp_options,
            key="front_timestamp_select",
        )
    else:
        selected_timestamp = None
        st.info("No saved session timestamps available for the selected system.")

    plot_df = system_df[system_df["trend_label"] == selected_trend].copy().sort_values("timestamp_dt")
    plot_df = plot_df.dropna(subset=["timestamp_dt", "value"])

    if plot_df.empty:
        st.warning("No data available for this selection.")
    else:
        latest = plot_df.iloc[-1]["value"]
        mean_val = plot_df["value"].mean()
        min_val = plot_df["value"].min()
        max_val = plot_df["value"].max()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest", f"{latest:.3f}")
        m2.metric("Mean", f"{mean_val:.3f}")
        m3.metric("Min", f"{min_val:.3f}")
        m4.metric("Max", f"{max_val:.3f}")

        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(plot_df["timestamp_dt"], plot_df["value"], marker="o")
        unit = plot_df["unit"].dropna().iloc[0] if not plot_df["unit"].dropna().empty else ""
        ax.set_title(f"{selected_trend} | {selected_system}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(f"Value ({unit})")
        ax.grid(True, alpha=0.3)
        add_reference_lines_ct(ax, selected_trend)
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Show trend data table"):
            st.dataframe(
                plot_df[
                    [
                        "timestamp",
                        "site_name",
                        "scanner_name",
                        "scanner_id",
                        "session_label",
                        "test_name",
                        "sequence_label",
                        "value",
                        "unit",
                        "status",
                        "details",
                    ]
                ],
                width="stretch",
            )

        csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download trend CSV",
            data=csv_bytes,
            file_name=f"{selected_trend}_{selected_system}_trend.csv".replace(" ", "_").replace("/", "_"),
            mime="text/csv",
            key="download_trend_csv",
        )

    st.subheader("Print selected session")

    if st.button("Generate PDF for selected session", key="front_selected_session_pdf"):
        if not selected_system or not selected_timestamp:
            st.error("Please select system and session timestamp.")
        else:
            session_df = build_single_session_df(front_trend_df, selected_system, selected_timestamp)

            if session_df.empty:
                st.warning("No data found for selected session.")
            else:
                pdf_path = build_single_session_pdf(session_df)

                with open(pdf_path, "rb") as f:
                    st.session_state.selected_session_pdf_bytes = f.read()
                    st.session_state.selected_session_pdf_name = pdf_path.name

                st.success(f"Selected session PDF created: {pdf_path.name}")

    if st.session_state.selected_session_pdf_bytes:
        st.download_button(
            "Download selected session PDF",
            data=st.session_state.selected_session_pdf_bytes,
            file_name=st.session_state.selected_session_pdf_name,
            mime="application/pdf",
            key="front_selected_session_pdf_download",
        )
