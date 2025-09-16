import os

# Define root directory (parent of config.py)
ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data paths
RAW_DATA_DIR = os.path.join(ROOT, "data", "raw")
UBPR_DATA_DIR = os.path.join(RAW_DATA_DIR, "ubpr_data")
CALL_REPORT_DATA_DIR = os.path.join(RAW_DATA_DIR, "CallReportData")
FAILED_BANK_LIST_DIR = os.path.join(RAW_DATA_DIR, "FailedBankList")

# Processed data paths
PROCESSED_DATA_DIR = os.path.join(ROOT, "data", "processed")
UBPR_PANEL_DIR = os.path.join(PROCESSED_DATA_DIR, "ubpr_panel")
CROSSWALK_DIR = os.path.join(PROCESSED_DATA_DIR, "crosswalk")

# Output paths
REPORTS_DIR = os.path.join(ROOT, "reports")

# Specific file paths
FAILED_BANK_LIST_FILE = os.path.join(FAILED_BANK_LIST_DIR, "FailedBankList.csv")
CROSSWALK_CSV = os.path.join(CROSSWALK_DIR, "rssd_cert_crosswalk.csv")
CROSSWALK_PARQUET = os.path.join(CROSSWALK_DIR, "rssd_cert_crosswalk.parquet")
UBPR_PANEL_FILE = os.path.join(UBPR_PANEL_DIR, "UBPR_Panel.parquet")
ANALYSIS_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "analysis_data.parquet")