import base64
from datetime import datetime
import io
import json
from pathlib import Path
import re
import time
import uuid

import matplotlib.pyplot as plt
import pandas as pd
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
import requests
import streamlit as st

APP_TITLE = "CT IEC Constancy QC Reporter"

# =========================================================
# GITHUB SECRETS INTEGRATION
# =========================================================
sec_github = st.secrets.get("github", {}) if hasattr(st, "secrets") else {}

DEFAULT_GITHUB_OWNER = sec_github.get("owner", "YOUR_GITHUB_USERNAME")
DEFAULT_GITHUB_REPO = sec_github.get("repo", "YOUR_REPO_NAME")
DEFAULT_GITHUB_BRANCH = sec_github.get("branch", "main")
DEFAULT_GITHUB_CSV_PATH = sec_github.get("path", "ct_qc_data/ct_qc_history.csv")
DEFAULT_GITHUB_TOKEN = sec_github.get("token", "")

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
    "lightmarker_start": r"(?mi)^\d+\s+Lightmarker\s*\(IEC\s*Constancy\)\s*$",
    "sag_cor_lightmarker_start": r"(?mi)^\d+\s+Sagittal/Coronal\s+Lightmarker\s*\(IEC\s*Constancy\)\s*$",
    "topogram_start": r"(?mi)^\d+\s+Topogram\s+Positioning\s*\(IEC\s*Constancy\)\s*$",
    "slice_start": r"(?mi)^\d+\s+Slice\s*\(IEC\s*Constancy\)\s*$",
    "homogeneity_start": r"(?mi)^\d+\s+Homogeneity\s*\(IEC\s*Constancy\)\s*$",
    "noise_start": r"(?mi)^\d+\s+Noise\s*\(IEC\s*Constancy\)\s*$",
    "mtf_start": r"(?mi)^\d+\s+MTF\s*\(IEC\s*Constancy\)\s*$",
    "contrast_start": r"(?mi)^\d+\s+Contrast\s*\(IEC\s*Constancy\)\s*$",
    "table_start": r"(?mi)^\d+\s+Table\s+Positioning\s*\(IEC\s*Constancy\)\s*$",
    "tube_start": r"(?mi)^\d+\s+Tube\s+Voltage\s*\(IEC\s*Constancy\)\s*$",
    "image_start": r"(?mi)^\d+\s+Image\s+Inspection\s*\(Constancy\)\s*$",
}

MODE_HEADER_PATTERNS = {
    "homogeneity": r"(?mi)^\d+\.3\.(\d+)\s+(.+)$",
    "noise": r"(?mi)^\d+\.3\.(\d+)\s+(.+)$",
    "mtf": r"(?mi)^\d+\.3\.(\d+)\s+(.+)$",
}

TOP_LEVEL_SECTION_RE = re.compile(r"(?mi)^\d+\s+.+$")

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
    return "\n".join(
        " ".join(line.replace("\u202f", " ").split())
        for line in str(text).splitlines()
    )


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
        "Lightmarker",
        "Sagittal Lightmarker",
        "Coronal Lightmarker",
        "Topogram Positioning",
        "Slice Thickness",
        "Water Value",
        "Homogeneity",
        "Noise",
        "MTF 50%",
        "MTF 10%",
        "Contrast Scale",
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

    out = out.sort_values(sort_cols, na_position="last").drop(
        columns=["_ct_order"]
    )
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
        r"^(.*?)\nIEC Constancy\nSerial Number:\s*([A-Za-z0-9-]+)(?:\n(.+))?$",
        clean,
        re.M,
    )
    if m_head_product:
        scanner_name = m_head_product.group(1).strip()
        serial_number = m_head_product.group(2).strip()
        if m_head_product.group(3):
            candidate = m_head_product.group(3).strip()
            if candidate and "page" not in candidate.lower():
                site_name = candidate

    m_prod = re.search(r"^Product Name\s+(.+?)\s*$", raw, re.I | re.M)
    if m_prod and m_prod.group(1).strip():
        scanner_name = " ".join(m_prod.group(1).split())

    m_serial = re.search(r"^Serial Number\s+(.+?)\s*$", raw, re.I | re.M)
    if m_serial and m_serial.group(1).strip():
        serial_number = " ".join(m_serial.group(1).split())

    m_hosp = re.search(r"^Hospital Name\s+(.+?)\s*$", raw, re.I | re.M)
    if m_hosp and m_hosp.group(1).strip():
        site_name = " ".join(m_hosp.group(1).split())

    m_customer = re.search(
        r"Customer\s+Name\s+(.+?)\s+Product\s+Name", clean, re.I | re.S
    )
    if not site_name and m_customer:
        candidate = " ".join(m_customer.group(1).split())
        if candidate and candidate.lower() not in {"-", "na"}:
            site_name = candidate

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

    m_ts = re.search(
        r"(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M)", clean, re.I
    )
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


def extract_section_by_title(text: str, title_regex: str) -> str:
    text = normalize_pdf_text(text)
    start_match = re.search(title_regex, text, flags=re.I | re.M)
    if not start_match:
        return ""

    start_idx = start_match.start()
    later_text = text[start_match.end() :]
    next_matches = list(TOP_LEVEL_SECTION_RE.finditer(later_text))

    if next_matches:
        end_idx = start_match.end() + next_matches[0].start()
        return text[start_idx:end_idx]

    return text[start_idx:]


def scan_top_level_sections(text: str):
    return re.findall(r"(?mi)^\d+\s+.+$", normalize_pdf_text(text))


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
        end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        )
        block_text = section_text[start:end]
        blocks.append((name, block_text))
    return blocks


def parse_tolerance_line(line):
    line = normalize_pdf_text(line)
    m = re.search(rf"Tolerance:\s*({NUM})\s*{TOL_SEP}\s*({NUM})", line, re.I)
    if m:
        return safe_float(m.group(1)), safe_float(m.group(2))

    cleaned = line.replace(":", ".")
    nums = [safe_float(x) for x in re.findall(NUM, cleaned)]
    if (
        ("Tolerance" in line or "Ref" in line or "..." in line or "–" in line)
        and len(nums) >= 2
    ):
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

    overall_status = (
        "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    )
    return worst, overall_status


def make_result(
    test_name, sequence_label, value, unit, criteria, status, details
):
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
        "LIGHTMARKER": SECTION_PATTERNS["lightmarker_start"],
        "SAG_COR": SECTION_PATTERNS["sag_cor_lightmarker_start"],
        "TOPOGRAM": SECTION_PATTERNS["topogram_start"],
        "SLICE": SECTION_PATTERNS["slice_start"],
        "HOMOGENEITY": SECTION_PATTERNS["homogeneity_start"],
        "NOISE": SECTION_PATTERNS["noise_start"],
        "MTF": SECTION_PATTERNS["mtf_start"],
        "CONTRAST": SECTION_PATTERNS["contrast_start"],
        "TABLE": SECTION_PATTERNS["table_start"],
        "TUBE": SECTION_PATTERNS["tube_start"],
        "IMAGE": SECTION_PATTERNS["image_start"],
    }

    with st.expander("DEBUG: section match dump"):
        st.write("Top-level sections found:")
        st.write(scan_top_level_sections(pdf_text))

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
def parse_ct_lightmarker(text):
    section = extract_section_by_title(
        text, SECTION_PATTERNS["lightmarker_start"]
    )
    results = []
    for line in iter_clean_lines(section):
        m = re.search(rf"^(.+?)\s+({NUM})\s+(OK|Pass|In Tol\.)$", line, re.I)
        if m:
            desc = m.group(1).strip()
            val = safe_float(m.group(2))
            status = (
                "PASS"
                if m.group(3).upper() in {"OK", "PASS", "IN TOL."}
                else "FAIL"
            )
            results.append(
                make_result(
                    "Lightmarker",
                    desc,
                    val,
                    "mm",
                    "Lightmarker within tolerance [-2.00, 2.00]",
                    status,
                    f"Mode: {desc}, Value: {val} mm",
                )
            )
    return results


def parse_ct_sag_cor_lightmarker(text):
    section = extract_section_by_title(
        text, SECTION_PATTERNS["sag_cor_lightmarker_start"]
    )
    results = []
    for line in iter_clean_lines(section):
        m = re.search(
            rf"^(.+?)\s+({NUM})\s+({NUM})\s+(OK|Pass|In Tol\.)$", line, re.I
        )
        if m:
            desc = m.group(1).strip()
            val_sag = safe_float(m.group(2))
            val_cor = safe_float(m.group(3))
            status = (
                "PASS"
                if m.group(4).upper() in {"OK", "PASS", "IN TOL."}
                else "FAIL"
            )
            results.append(
                make_result(
                    "Sagittal Lightmarker",
                    desc,
                    abs(val_sag) if val_sag is not None else None,
                    "mm",
                    "Sagittal lightmarker within tolerance",
                    status,
                    f"Sagittal deviation: {val_sag} mm",
                )
            )
            results.append(
                make_result(
                    "Coronal Lightmarker",
                    desc,
                    abs(val_cor) if val_cor is not None else None,
                    "mm",
                    "Coronal lightmarker within tolerance",
                    status,
                    f"Coronal deviation: {val_cor} mm",
                )
            )
    return results


def parse_ct_topogram_positioning(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["topogram_start"])
    results = []
    for line in iter_clean_lines(section):
        m = re.search(rf"^(.+?)\s+({NUM})\s+(OK|Pass|In Tol\.)$", line, re.I)
        if m:
            desc = m.group(1).strip()
            val = safe_float(m.group(2))
            status = (
                "PASS"
                if m.group(3).upper() in {"OK", "PASS", "IN TOL."}
                else "FAIL"
            )
            results.append(
                make_result(
                    "Topogram Positioning",
                    desc,
                    val,
                    "mm",
                    "Topogram positioning within tolerance",
                    status,
                    f"Deviation: {val} mm",
                )
            )
    return results


def parse_ct_slice(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["slice_start"])
    if not section:
        return []
    results = []
    sub_blocks = re.split(r"(?mi)^\d+\.\d+\.\d+\s+", section)
    for block in sub_blocks:
        lines = list(iter_clean_lines(block))
        if not lines:
            continue
        seq_label = lines[0]
        rows = []
        for line in lines[1:]:
            m = re.search(
                rf"^(\d+)\s+({NUM})\s+(In Tol\.|Out Tol\.|OK|Pass|Fail)$",
                line,
                re.I,
            )
            if m:
                slice_no = int(m.group(1))
                val = safe_float(m.group(2))
                status = (
                    "PASS"
                    if "IN TOL" in m.group(3).upper()
                    or m.group(3).upper() in {"OK", "PASS"}
                    else "FAIL"
                )
                rows.append({"slice": slice_no, "value": val, "status": status})
        if rows:
            worst, overall_status = summarize_slice_rows(
                rows, "Slice Thickness", worst_by="max_abs"
            )
            slice_summary = "; ".join(
                f"S{r['slice']}={format_num(r['value'])}mm {r['status']}"
                for r in rows
            )
            results.append(
                make_result(
                    "Slice Thickness",
                    seq_label,
                    worst["value"],
                    "mm",
                    "Slice thickness within tolerance",
                    overall_status,
                    f"Worst slice {worst['slice']} thickness = {format_num(worst['value'])} mm. All slices: {slice_summary}",
                )
            )
    return results


def parse_ct_water_value_and_homogeneity(text):
    section = extract_section_by_title(
        text, SECTION_PATTERNS["homogeneity_start"]
    )
    mode_blocks = split_mode_blocks(section, MODE_HEADER_PATTERNS["homogeneity"])

    results = []
    for mode_name, block in mode_blocks:
        lines = list(iter_clean_lines(block))

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
                row = {
                    "slice": int(m_inline.group(1)),
                    "value": safe_float(m_inline.group(2)),
                    "reference": safe_float(m_inline.group(3)),
                    "low": safe_float(m_inline.group(4)),
                    "high": safe_float(m_inline.group(5)),
                }
                row["status"] = row_status(row["value"], row["low"], row["high"])
                water_rows.append(row)
                continue

            m_simple = re.search(rf"^(\d+)\s+({NUM})\s+({NUM})$", line, re.I)
            if (
                m_simple
                and current_low is not None
                and current_high is not None
            ):
                row = {
                    "slice": int(m_simple.group(1)),
                    "value": safe_float(m_simple.group(2)),
                    "reference": safe_float(m_simple.group(3)),
                    "low": current_low,
                    "high": current_high,
                }
                row["status"] = row_status(row["value"], row["low"], row["high"])
                water_rows.append(row)

        if water_rows:
            worst, overall_status = summarize_slice_rows(
                water_rows, "Water Value", worst_by="max_abs"
            )
            slice_summary = "; ".join(
                f"S{r['slice']}={format_num(r['value'])} HU [{format_num(r['low'], 2)},{format_num(r['high'], 2)}] {r['status']}"
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
                    f"tolerance [{format_num(worst['low'], 2)},{format_num(worst['high'], 2)}]. "
                    f"All slices: {slice_summary}",
                )
            )

        diff_rows = []
        current_slice = None
        current_low, current_high = -4.0, 4.0

        for line in lines:
            m_slice = re.match(r"^Slice\s+(\d+)$", line, re.I)
            if m_slice:
                current_slice = int(m_slice.group(1))
                continue

            low, high = parse_tolerance_line(line)
            if low is not None and high is not None:
                current_low, current_high = low, high
                continue

            if re.search(r"^Center", line, re.I):
                continue

            m_diff_inline = re.search(
                rf"^(Diff\.\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})$",
                line,
                re.I,
            )
            if m_diff_inline and current_slice is not None:
                row = {
                    "slice": current_slice,
                    "position": m_diff_inline.group(1),
                    "value": safe_float(m_diff_inline.group(2)),
                    "reference": safe_float(m_diff_inline.group(3)),
                    "low": safe_float(m_diff_inline.group(4)),
                    "high": safe_float(m_diff_inline.group(5)),
                }
                row["status"] = row_status(row["value"], row["low"], row["high"])
                diff_rows.append(row)
                continue

            m_diff_simple = re.search(
                rf"^(Diff\.\d+)\s+({NUM})\s+({NUM})$", line, re.I
            )
            if m_diff_simple and current_slice is not None:
                row = {
                    "slice": current_slice,
                    "position": m_diff_simple.group(1),
                    "value": safe_float(m_diff_simple.group(2)),
                    "reference": safe_float(m_diff_simple.group(3)),
                    "low": (
                        current_low if current_low is not None else -4.0
                    ),
                    "high": (
                        current_high if current_high is not None else 4.0
                    ),
                }
                row["status"] = row_status(row["value"], row["low"], row["high"])
                diff_rows.append(row)

        if diff_rows:
            worst, overall_status = summarize_slice_rows(
                diff_rows, "Homogeneity", worst_by="max_abs"
            )
            slice_summary = "; ".join(
                f"S{r['slice']} {r['position']}={format_num(r['value'])} HU [{format_num(r['low'], 2)},{format_num(r['high'], 2)}] {r['status']}"
                for r in diff_rows
            )
            results.append(
                make_result(
                    "Homogeneity",
                    mode_name,
                    abs(worst["value"]),
                    "HU",
                    "Max peripheral-centre difference within tolerance",
                    overall_status,
                    f"Worst slice {worst['slice']} {worst['position']} = {format_num(worst['value'])} HU; "
                    f"tolerance [{format_num(worst['low'], 2)},{format_num(worst['high'], 2)}]. "
                    f"All differences: {slice_summary}",
                )
            )

    return results


def parse_ct_noise(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["noise_start"])
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

        worst, overall_status = summarize_slice_rows(
            rows, "Noise", worst_by="max"
        )
        slice_summary = "; ".join(
            f"S{r['slice']}={format_num(r['value'])} HU [{format_num(r['low'], 2)},{format_num(r['high'], 2)}] {r['status']}"
            for r in rows
        )
        results.append(
            make_result(
                "Noise",
                mode_name,
                worst["value"],
                "HU",
                "Each slice within slice-specific tolerance",
                overall_status,
                f"Worst slice {worst['slice']} noise = {format_num(worst['value'])} HU; "
                f"reference {format_num(worst['reference'])} HU; "
                f"tolerance [{format_num(worst['low'], 2)},{format_num(worst['high'], 2)}]. "
                f"All slices: {slice_summary}",
            )
        )

    return results


def parse_ct_mtf(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["mtf_start"])
    mode_blocks = split_mode_blocks(section, MODE_HEADER_PATTERNS["mtf"])

    results = []
    for mode_name, block in mode_blocks:
        lines = list(iter_clean_lines(block))

        tol50 = (None, None)
        tol10 = (None, None)
        ref50 = None
        ref10 = None
        rows50 = []
        rows10 = []

        i = 0
        while i < len(lines):
            line = lines[i]
            m_tol_combined = re.search(
                rf"Tolerance:\s*({NUM})\s*{TOL_SEP}\s*({NUM})\s+Reference\s+Tolerance:\s*({NUM})\s*{TOL_SEP}\s*({NUM})",
                line,
                re.I,
            )
            if m_tol_combined:
                tol50 = (
                    safe_float(m_tol_combined.group(1)),
                    safe_float(m_tol_combined.group(2)),
                )
                tol10 = (
                    safe_float(m_tol_combined.group(3)),
                    safe_float(m_tol_combined.group(4)),
                )
                i += 1
                continue

            m_ref = re.search(rf"^Reference:\s*({NUM})$", line, re.I)
            if m_ref:
                first_ref = safe_float(m_ref.group(1))
                low1, high1 = (None, None)
                if i + 1 < len(lines):
                    low1, high1 = parse_tolerance_line(lines[i + 1])
                m_ref2 = None
                if i + 2 < len(lines):
                    m_ref2 = re.search(
                        rf"^Reference:\s*({NUM})$", lines[i + 2], re.I
                    )
                low2, high2 = (None, None)
                if i + 3 < len(lines):
                    low2, high2 = parse_tolerance_line(lines[i + 3])

                if (
                    low1 is not None
                    and high1 is not None
                    and m_ref2
                    and low2 is not None
                    and high2 is not None
                ):
                    ref50 = first_ref
                    tol50 = (low1, high1)
                    ref10 = safe_float(m_ref2.group(1))
                    tol10 = (low2, high2)
                    i += 4
                    continue

            m_row = re.search(
                rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})$", line, re.I
            )
            if m_row:
                slice_no = int(m_row.group(1))
                val50 = safe_float(m_row.group(2))
                ref50_row = safe_float(m_row.group(3))
                val10 = safe_float(m_row.group(4))
                ref10_row = safe_float(m_row.group(5))

                row50 = {
                    "slice": slice_no,
                    "value": val50,
                    "reference": ref50_row,
                    "low": tol50[0],
                    "high": tol50[1],
                }
                row50["status"] = row_status(
                    row50["value"], row50["low"], row50["high"]
                )
                rows50.append(row50)

                row10 = {
                    "slice": slice_no,
                    "value": val10,
                    "reference": ref10_row,
                    "low": tol10[0],
                    "high": tol10[1],
                }
                row10["status"] = row_status(
                    row10["value"], row10["low"], row10["high"]
                )
                rows10.append(row10)
                i += 1
                continue

            m_sharp = re.search(rf"^(\d+)\s+({NUM})\s+({NUM})$", line, re.I)
            if m_sharp and tol50[0] is not None and tol10[0] is not None:
                slice_no = int(m_sharp.group(1))
                val50 = safe_float(m_sharp.group(2))
                val10 = safe_float(m_sharp.group(3))

                row50 = {
                    "slice": slice_no,
                    "value": val50,
                    "reference": ref50,
                    "low": tol50[0],
                    "high": tol50[1],
                }
                row50["status"] = row_status(
                    row50["value"], row50["low"], row50["high"]
                )
                rows50.append(row50)

                row10 = {
                    "slice": slice_no,
                    "value": val10,
                    "reference": ref10,
                    "low": tol10[0],
                    "high": tol10[1],
                }
                row10["status"] = row_status(
                    row10["value"], row10["low"], row10["high"]
                )
                rows10.append(row10)
                i += 1
                continue

            i += 1

        if rows50:
            worst50, status50 = summarize_slice_rows(
                rows50, "MTF 50%", worst_by="min"
            )
            slice_summary50 = "; ".join(
                f"S{r['slice']}={format_num(r['value'])} ref={format_num(r['reference'])} [{format_num(r['low'], 2)},{format_num(r['high'], 2)}] {r['status']}"
                for r in rows50
            )
            results.append(
                make_result(
                    "MTF 50%",
                    mode_name,
                    worst50["value"],
                    "lp/cm",
                    "All slices within MTF 50% tolerance",
                    status50,
                    f"Worst slice {worst50['slice']} MTF 50% = {format_num(worst50['value'])} lp/cm; "
                    f"reference {format_num(worst50['reference'])} lp/cm; "
                    f"tolerance [{format_num(worst50['low'], 2)},{format_num(worst50['high'], 2)}]. "
                    f"All slices: {slice_summary50}",
                )
            )

        if rows10:
            worst10, status10 = summarize_slice_rows(
                rows10, "MTF 10%", worst_by="min"
            )
            slice_summary10 = "; ".join(
                f"S{r['slice']}={format_num(r['value'])} ref={format_num(r['reference'])} [{format_num(r['low'], 2)},{format_num(r['high'], 2)}] {r['status']}"
                for r in rows10
            )
            results.append(
                make_result(
                    "MTF 10%",
                    mode_name,
                    worst10["value"],
                    "lp/cm",
                    "All slices within MTF 10% tolerance",
                    status10,
                    f"Worst slice {worst10['slice']} MTF 10% = {format_num(worst10['value'])} lp/cm; "
                    f"reference {format_num(worst10['reference'])} lp/cm; "
                    f"tolerance [{format_num(worst10['low'], 2)},{format_num(worst10['high'], 2)}]. "
                    f"All slices: {slice_summary10}",
                )
            )

    return results


def parse_ct_contrast(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["contrast_start"])
    if not section:
        return []
    rows = []
    for line in iter_clean_lines(section):
        m = re.search(
            rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+(In Tol\.|Out Tol\.|OK|Pass|Fail)$",
            line,
            re.I,
        )
        if m:
            slice_no = int(m.group(1))
            contrast_scale = safe_float(m.group(4))
            status = (
                "PASS"
                if "IN TOL" in m.group(5).upper()
                or m.group(5).upper() in {"OK", "PASS"}
                else "FAIL"
            )
            rows.append(
                {
                    "slice": slice_no,
                    "contrast_scale": contrast_scale,
                    "status": status,
                }
            )
    if rows:
        worst = max(
            rows,
            key=lambda x: abs(x["contrast_scale"])
            if x["contrast_scale"] is not None
            else -1,
        )
        overall_status = (
            "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
        )
        return [
            make_result(
                "Contrast Scale",
                "Typical body",
                worst["contrast_scale"],
                "1/HUcm",
                "Contrast scale within tolerance",
                overall_status,
                f"Worst slice {worst['slice']} contrast scale = {format_num(worst['contrast_scale'], 7)}",
            )
        ]
    return []


def parse_ct_table_positioning(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["table_start"])
    nominal_map = {1: -300.0, 2: 0.0, 3: 300.0, 4: 0.0}

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
        nominal = nominal_map.get(pos)

        cont_measured = safe_float(m.group(2))
        cont_low = safe_float(m.group(3))
        cont_high = safe_float(m.group(4))

        step_measured = safe_float(m.group(5))
        step_low = safe_float(m.group(6))
        step_high = safe_float(m.group(7))

        cont_dev = cont_measured - nominal if nominal is not None else None
        step_dev = step_measured - nominal if nominal is not None else None

        rowc = {
            "position": pos,
            "nominal": nominal,
            "measured": cont_measured,
            "value": abs(cont_dev) if cont_dev is not None else None,
            "deviation": cont_dev,
            "low": cont_low,
            "high": cont_high,
        }
        rowc["status"] = row_status(rowc["measured"], rowc["low"], rowc["high"])
        cont_rows.append(rowc)

        rows = {
            "position": pos,
            "nominal": nominal,
            "measured": step_measured,
            "value": abs(step_dev) if step_dev is not None else None,
            "deviation": step_dev,
            "low": step_low,
            "high": step_high,
        }
        rows["status"] = row_status(rows["measured"], rows["low"], rows["high"])
        step_rows.append(rows)

    results = []
    if cont_rows:
        worst = max(
            cont_rows,
            key=lambda x: abs(x["deviation"])
            if x["deviation"] is not None
            else -1,
        )
        overall_status = (
            "PASS" if all(r["status"] == "PASS" for r in cont_rows) else "FAIL"
        )
        pos_summary = "; ".join(
            f"P{r['position']}: nominal={format_num(r['nominal'])} mm, "
            f"measured={format_num(r['measured'])} mm, "
            f"Δ={format_num(r['deviation'])} mm, "
            f"allowed [{format_num(r['low'], 2)},{format_num(r['high'], 2)}], {r['status']}"
            for r in cont_rows
        )
        results.append(
            make_result(
                "Table Positioning (Continuous)",
                "Continuous movement",
                abs(worst["deviation"]),
                "mm",
                "At each of 4 positions within tolerance",
                overall_status,
                f"Worst position P{worst['position']} deviation = {format_num(worst['deviation'])} mm; "
                f"nominal {format_num(worst['nominal'])} mm; measured {format_num(worst['measured'])} mm; "
                f"tolerance [{format_num(worst['low'], 2)}, {format_num(worst['high'], 2)}]. "
                f"All positions: {pos_summary}",
            )
        )

    if step_rows:
        worst = max(
            step_rows,
            key=lambda x: abs(x["deviation"])
            if x["deviation"] is not None
            else -1,
        )
        overall_status = (
            "PASS" if all(r["status"] == "PASS" for r in step_rows) else "FAIL"
        )
        pos_summary = "; ".join(
            f"P{r['position']}: nominal={format_num(r['nominal'])} mm, "
            f"measured={format_num(r['measured'])} mm, "
            f"Δ={format_num(r['deviation'])} mm, "
            f"allowed [{format_num(r['low'], 2)},{format_num(r['high'], 2)}], {r['status']}"
            for r in step_rows
        )
        results.append(
            make_result(
                "Table Positioning (Stepwise)",
                "Stepwise movement",
                abs(worst["deviation"]),
                "mm",
                "At each of 4 positions within tolerance",
                overall_status,
                f"Worst position P{worst['position']} deviation = {format_num(worst['deviation'])} mm; "
                f"nominal {format_num(worst['nominal'])} mm; measured {format_num(worst['measured'])} mm; "
                f"tolerance [{format_num(worst['low'], 2)}, {format_num(worst['high'], 2)}]. "
                f"All positions: {pos_summary}",
            )
        )

    return results


def parse_ct_tube_voltage(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["tube_start"])
    rows = []
    for line in iter_clean_lines(section):
        m = re.search(
            rf"^({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*{TOL_SEP}\s*({NUM})$",
            line,
            re.I,
        )
        if not m:
            continue
        measured = safe_float(m.group(1))
        reference = safe_float(m.group(2))
        low = safe_float(m.group(4))
        high = safe_float(m.group(5))
        dev = measured - reference if (measured and reference) else None
        row = {
            "measured": measured,
            "reference": reference,
            "deviation": dev,
            "value": abs(dev) if dev is not None else None,
            "low": low,
            "high": high,
        }
        row["status"] = row_status(row["measured"], row["low"], row["high"])
        rows.append(row)

    if not rows:
        return []

    worst = max(
        rows,
        key=lambda x: abs(x["deviation"]) if x["deviation"] is not None else -1,
    )
    overall_status = (
        "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    )
    return [
        make_result(
            "Tube Voltage",
            "Constancy",
            abs(worst["deviation"]),
            "kV",
            "Tube voltage within tolerance",
            overall_status,
            f"Measured = {format_num(worst['measured'])} kV, "
            f"Reference = {format_num(worst['reference'])} kV, "
            f"Deviation = {format_num(worst['deviation'])} kV, "
            f"Tolerance [{format_num(worst['low'], 2)}, {format_num(worst['high'], 2)}]",
        )
    ]


def parse_ct_image_inspection(text):
    section = extract_section_by_title(text, SECTION_PATTERNS["image_start"])
    status = "PASS"
    details = "Image inspection completed successfully."
    for line in iter_clean_lines(section):
        if "fail" in line.lower():
            status = "FAIL"
            details = line
        elif "accept" in line.lower() or "pass" in line.lower():
            details = line

    return [
        make_result(
            "Image Inspection",
            "Visual",
            1.0 if status == "PASS" else 0.0,
            "",
            "Visual image inspection passed",
            status,
            details,
        )
    ]


def parse_ct_pdf(uploaded_file):
    text = read_pdf_text(uploaded_file)
    meta = extract_pdf_metadata(text)

    results = []
    results.extend(parse_ct_lightmarker(text))
    results.extend(parse_ct_sag_cor_lightmarker(text))
    results.extend(parse_ct_topogram_positioning(text))
    results.extend(parse_ct_slice(text))
    results.extend(parse_ct_water_value_and_homogeneity(text))
    results.extend(parse_ct_noise(text))
    results.extend(parse_ct_mtf(text))
    results.extend(parse_ct_contrast(text))
    results.extend(parse_ct_table_positioning(text))
    results.extend(parse_ct_tube_voltage(text))
    results.extend(parse_ct_image_inspection(text))

    return meta, results, text


# =========================================================
# GITHUB & STORAGE
# =========================================================
def load_local_history() -> pd.DataFrame:
    if LOCAL_HISTORY_CSV.exists():
        try:
            df = pd.read_csv(LOCAL_HISTORY_CSV)
            return normalize_history_df(df)
        except Exception:
            pass
    return empty_history_df()


def save_local_history(df: pd.DataFrame):
    df = normalize_history_df(df)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(LOCAL_HISTORY_CSV, index=False)


def fetch_github_csv(cfg) -> pd.DataFrame:
    if not github_is_ready(cfg):
        return load_local_history()

    url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}/contents/{cfg['path']}"
    headers = {"Authorization": f"Bearer {cfg['token']}"}
    params = {"ref": cfg.get("branch", "main")}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            content = resp.json()
            file_content = base64.b64decode(content["content"]).decode("utf-8")
            df = pd.read_csv(io.StringIO(file_content))
            return normalize_history_df(df)
    except Exception:
        pass
    return load_local_history()


def push_github_csv(cfg, df: pd.DataFrame, commit_message: str) -> bool:
    if not github_is_ready(cfg):
        return False

    df = normalize_history_df(df)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    encoded_content = base64.b64encode(csv_bytes).decode("utf-8")

    url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}/contents/{cfg['path']}"
    headers = {"Authorization": f"Bearer {cfg['token']}"}

    try:
        get_resp = requests.get(
            url, headers=headers, params={"ref": cfg.get("branch", "main")}
        )
        sha = None
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")

        payload = {
            "message": commit_message,
            "content": encoded_content,
            "branch": cfg.get("branch", "main"),
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(url, headers=headers, json=payload, timeout=15)
        return put_resp.status_code in {200, 201}
    except Exception:
        return False


# =========================================================
# REPORT GENERATION & PLOTTING
# =========================================================
def generate_trend_plot(history_df, scanner_id, test_name):
    df = normalize_history_df(history_df)
    sub = df[
        (df["scanner_id"] == scanner_id) & (df["test_name"] == test_name)
    ].copy()
    if sub.empty:
        return None

    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(
        sub["timestamp"],
        sub["value"],
        marker="o",
        linestyle="-",
        color="#1f77b4",
    )
    ax.set_title(f"Trend: {test_name}", fontsize=10)
    ax.set_ylabel(sub["unit"].iloc[0] or "", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, fontsize=8)
    plt.tight_layout()

    path = CHARTS_DIR / f"trend_{scanner_id}_{sanitize_filename(test_name)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_pdf_report(session_df, output_path, logo_path=None):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    cell_style = ParagraphStyle(
        "CellText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )

    story = []

    if logo_path and Path(logo_path).exists():
        story.append(RLImage(str(logo_path), width=120, height=40))
        story.append(Spacer(1, 10))

    story.append(Paragraph(APP_TITLE, title_style))
    story.append(Paragraph("IEC Constancy Quality Control Report", meta_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#cbd5e1"),
            spaceAfter=15,
        )
    )

    if not session_df.empty:
        row0 = session_df.iloc[0]
        meta_text = f"""
        <b>Site Name:</b> {row0.get('site_name', '')}<br/>
        <b>Scanner:</b> {row0.get('scanner_name', '')}<br/>
        <b>Session Timestamp:</b> {row0.get('timestamp', '')}<br/>
        <b>Session Label:</b> {row0.get('session_label', '')}
        """
        story.append(Paragraph(meta_text, meta_style))
        story.append(Spacer(1, 15))

        table_data = [
            [
                Paragraph("<b>Test</b>", cell_style),
                Paragraph("<b>Sequence / Mode</b>", cell_style),
                Paragraph("<b>Value</b>", cell_style),
                Paragraph("<b>Status</b>", cell_style),
                Paragraph("<b>Details</b>", cell_style),
            ]
        ]

        for _, r in session_df.iterrows():
            status_color = (
                "#16a34a" if str(r["status"]).upper() == "PASS" else "#dc2626"
            )
            status_p = Paragraph(
                f'<font color="{status_color}"><b>{r["status"]}</b></font>',
                cell_style,
            )
            table_data.append(
                [
                    Paragraph(str(r["test_name"]), cell_style),
                    Paragraph(str(r["sequence_label"]), cell_style),
                    Paragraph(
                        f"{format_num(r['value'])} {r['unit']}".strip(),
                        cell_style,
                    ),
                    status_p,
                    Paragraph(str(r["details"]), cell_style),
                ]
            )

        t = Table(table_data, colWidths=[100, 90, 60, 50, 240])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#cbd5e1"),
                    ),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(t)

    doc.build(story)
    return output_path


# =========================================================
# STREAMLIT UI
# =========================================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Configuration & GitHub Sync")
        github_token = st.text_input(
            "GitHub Token", value=DEFAULT_GITHUB_TOKEN, type="password"
        )
        github_owner = st.text_input(
            "GitHub Owner", value=DEFAULT_GITHUB_OWNER
        )
        github_repo = st.text_input("GitHub Repo", value=DEFAULT_GITHUB_REPO)
        github_branch = st.text_input(
            "GitHub Branch", value=DEFAULT_GITHUB_BRANCH
        )
        github_path = st.text_input(
            "CSV Path in Repo", value=DEFAULT_GITHUB_CSV_PATH
        )

        cfg = {
            "token": github_token,
            "owner": github_owner,
            "repo": github_repo,
            "branch": github_branch,
            "path": github_path,
        }

        st.markdown("---")
        if st.button("Fetch History from GitHub"):
            df_fetched = fetch_github_csv(cfg)
            save_local_history(df_fetched)
            st.success(f"Loaded {len(df_fetched)} records.")

    history_df = load_local_history()

    tab_upload, tab_dashboard, tab_history = st.tabs(
        ["Upload & Parse", "Dashboard & Trends", "History Database"]
    )

    with tab_upload:
        st.subheader("Upload CT QC PDF Report")
        uploaded_file = st.file_uploader("Choose CT QC PDF", type=["pdf"])

        col1, col2 = st.columns(2)
        with col1:
            site_input = st.text_input("Site Name (Override)")
        with col2:
            scanner_input = st.text_input("Scanner Name (Override)")

        session_label = st.text_input(
            "Session Label / Notes", value="Routine QC"
        )

        if uploaded_file is not None:
            meta, results, pdf_text = parse_ct_pdf(uploaded_file)

            final_site = (
                site_input.strip() or meta["site_name"] or "Unknown Site"
            )
            final_scanner = (
                scanner_input.strip()
                or meta["scanner_name"]
                or "Unknown Scanner"
            )
            scanner_id = build_scanner_id(final_site, final_scanner)
            timestamp = meta["timestamp_iso"] or datetime.now().isoformat(
                timespec="seconds"
            )

            st.markdown("### Extracted Metadata")
            st.json(
                {
                    "Site Name": final_site,
                    "Scanner Name": final_scanner,
                    "Serial Number": meta["serial_number"],
                    "Timestamp": timestamp,
                }
            )

            st.markdown("### Extracted QC Results")
            res_df = pd.DataFrame(results)
            if not res_df.empty:
                st.dataframe(res_df, use_container_width=True)

                if st.button("Commit Results to History"):
                    new_rows = []
                    for r in results:
                        new_rows.append(
                            {
                                "timestamp": timestamp,
                                "session_label": session_label,
                                "site_name": final_site,
                                "scanner_name": final_scanner,
                                "scanner_id": scanner_id,
                                "test_name": r["test_name"],
                                "value": r["value"],
                                "unit": r["unit"],
                                "criteria": r["criteria"],
                                "status": r["status"],
                                "details": r["details"],
                                "source_file": uploaded_file.name,
                                "sequence_label": r["sequence_label"],
                            }
                        )

                    new_df = pd.DataFrame(new_rows)
                    updated_history = pd.concat(
                        [history_df, new_df], ignore_index=True
                    )
                    updated_history = normalize_history_df(updated_history)
                    save_local_history(updated_history)

                    if github_is_ready(cfg):
                        success = push_github_csv(
                            cfg,
                            updated_history,
                            f"Add QC report for {final_scanner} at {final_site}",
                        )
                        if success:
                            st.success("Successfully pushed to GitHub!")
                        else:
                            st.warning(
                                "Saved locally, but GitHub push failed."
                            )
                    else:
                        st.success("Saved successfully to local storage.")
            else:
                st.warning("No QC results extracted from PDF.")

            if DEBUG_MODE:
                debug_dump_sections(pdf_text)

    with tab_dashboard:
        st.subheader("QC Dashboard & Trends")
        history_df = load_local_history()

        if history_df.empty:
            st.info("No history records available. Upload a PDF report first.")
        else:
            scanners = history_df[
                ["scanner_id", "site_name", "scanner_name"]
            ].drop_duplicates()
            scanner_options = {
                f"{r.site_name} - {r.scanner_name} ({r.scanner_id})": r.scanner_id
                for _, r in scanners.iterrows()
            }

            selected_label = st.selectbox(
                "Select Scanner", list(scanner_options.keys())
            )
            selected_scanner_id = scanner_options[selected_label]

            scanner_data = history_df[
                history_df["scanner_id"] == selected_scanner_id
            ]
            timestamps = sorted(
                scanner_data["timestamp"].unique(), reverse=True
            )

            if timestamps:
                selected_ts = st.selectbox(
                    "Select Session Timestamp", timestamps
                )
                session_df = build_single_session_df(
                    history_df, selected_scanner_id, selected_ts
                )

                st.markdown("### Session Summary")
                st.dataframe(session_df, use_container_width=True)

                if st.button("Generate PDF Report"):
                    report_filename = (
                        REPORTS_DIR
                        / f"QC_Report_{sanitize_filename(selected_scanner_id)}_{sanitize_filename(selected_ts)}.pdf"
                    )
                    generate_pdf_report(
                        session_df,
                        report_filename,
                        logo_path=LOGO_PATH if LOGO_PATH.exists() else None,
                    )
                    with open(report_filename, "rb") as f:
                        st.download_button(
                            "Download PDF Report",
                            f,
                            file_name=report_filename.name,
                            mime="application/pdf",
                        )

                st.markdown("### Parameter Trends")
                available_tests = sorted(scanner_data["test_name"].unique())
                selected_test = st.selectbox(
                    "Select Test for Trend Analysis", available_tests
                )
                trend_path = generate_trend_plot(
                    history_df, selected_scanner_id, selected_test
                )
                if trend_path and trend_path.exists():
                    st.image(str(trend_path))

    with tab_history:
        st.subheader("Complete History Database")
        history_df = load_local_history()
        st.dataframe(history_df, use_container_width=True)

        if not history_df.empty:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full History CSV",
                csv_data,
                file_name="ct_qc_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
