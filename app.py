import re
import io
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "CT Catphan600 QC Reporter"

DATA_DIR = Path("ct_qc_data")
LOCAL_HISTORY_CSV = DATA_DIR / "ct_qc_history.csv"

DATA_DIR.mkdir(exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def safe_float(x):
    try:
        return float(x)
    except:
        return None


def build_scanner_id(site, scanner):
    return f"{site}_{scanner}".replace(" ", "_").lower()


def get_columns():
    return [
        "timestamp",
        "site_name",
        "scanner_name",
        "scanner_id",
        "test_name",
        "value",
        "unit",
        "criteria",
        "status",
        "details",
    ]


def empty_df():
    return pd.DataFrame(columns=get_columns())


def normalize(df):
    if df is None or df.empty:
        return empty_df()

    for col in get_columns():
        if col not in df.columns:
            df[col] = None

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[get_columns()]


# =========================================================
# SAFE LOAD HISTORY (FIXED)
# =========================================================
def load_history():
    if LOCAL_HISTORY_CSV.exists():
        try:
            if LOCAL_HISTORY_CSV.stat().st_size == 0:
                return empty_df()
            return normalize(pd.read_csv(LOCAL_HISTORY_CSV))
        except:
            return empty_df()
    return empty_df()


def save_history(df):
    normalize(df).to_csv(LOCAL_HISTORY_CSV, index=False)


# =========================================================
# CT PARSERS (CATPHAN)
# =========================================================

def parse_uniformity(text):
    m = re.search(r"Uniformity:\s*([0-9.\-]+)", text)
    val = safe_float(m.group(1)) if m else None

    return {
        "test_name": "Uniformity",
        "value": val,
        "unit": "HU",
        "criteria": "±5 HU",
        "status": "PASS" if val and abs(val) <= 5 else "FAIL",
        "details": f"Uniformity = {val} HU",
    }


def parse_noise(text):
    m = re.search(r"Noise:\s*([0-9.\-]+)", text)
    val = safe_float(m.group(1)) if m else None

    return {
        "test_name": "Noise",
        "value": val,
        "unit": "HU",
        "criteria": "< 10 HU",
        "status": "PASS" if val and val < 10 else "FAIL",
        "details": f"Noise = {val} HU",
    }


def parse_ct_number(text):
    m = re.search(r"Water CT Number:\s*([0-9.\-]+)", text)
    val = safe_float(m.group(1)) if m else None

    return {
        "test_name": "CT Number Accuracy",
        "value": val,
        "unit": "HU",
        "criteria": "0 ± 5 HU",
        "status": "PASS" if val and abs(val) <= 5 else "FAIL",
        "details": f"Water CT = {val} HU",
    }


def parse_mtf(text):
    m = re.search(r"MTF 50%:\s*([0-9.\-]+)", text)
    val = safe_float(m.group(1)) if m else None

    return {
        "test_name": "MTF 50%",
        "value": val,
        "unit": "lp/cm",
        "criteria": "Trend",
        "status": "PASS" if val else "FAIL",
        "details": f"MTF50 = {val}",
    }


def infer_parser(text):
    if "Uniformity" in text:
        return parse_uniformity(text)
    if "Noise" in text:
        return parse_noise(text)
    if "Water CT" in text:
        return parse_ct_number(text)
    if "MTF" in text:
        return parse_mtf(text)

    return {
        "test_name": "Unknown",
        "value": None,
        "unit": "",
        "criteria": "",
        "status": "FAIL",
        "details": "Could not parse",
    }


# =========================================================
# APP
# =========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Sidebar
with st.sidebar:
    site = st.text_input("Site")
    scanner = st.text_input("Scanner")
    session_label = st.text_input("Session", "Daily QC")

scanner_id = build_scanner_id(site, scanner)
timestamp = datetime.now().isoformat()

uploaded_files = st.file_uploader(
    "Upload CT result files", type=["txt"], accept_multiple_files=True
)

history_df = load_history()

results = []

# =========================================================
# PARSE FILES
# =========================================================
if uploaded_files:
    for f in uploaded_files:
        text = f.read().decode("utf-8", errors="ignore")
        res = infer_parser(text)
        results.append(res)

    df = pd.DataFrame(results)
    st.subheader("Results")
    st.dataframe(df, width="stretch")

    overall = "PASS" if (df["status"] == "PASS").all() else "FAIL"
    st.metric("Overall", overall)

    # SAVE
    if st.button("Save Session"):
        rows = []
        for r in results:
            rows.append(
                {
                    "timestamp": timestamp,
                    "site_name": site,
                    "scanner_name": scanner,
                    "scanner_id": scanner_id,
                    "test_name": r["test_name"],
                    "value": r["value"],
                    "unit": r["unit"],
                    "criteria": r["criteria"],
                    "status": r["status"],
                    "details": r["details"],
                }
            )

        history_df = pd.concat([history_df, pd.DataFrame(rows)])
        save_history(history_df)

        st.success("Saved to history")


# =========================================================
# TREND
# =========================================================
st.subheader("Trend")

if not history_df.empty:
    tests = history_df["test_name"].unique().tolist()

    selected_test = st.selectbox("Select test", tests)

    df = history_df[history_df["test_name"] == selected_test].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["value"], marker="o")
    ax.set_title(selected_test)
    ax.grid(True)

    st.pyplot(fig)
else:
    st.info("No history yet")
