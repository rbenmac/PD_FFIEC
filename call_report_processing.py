import pandas as pd
import os
from functools import reduce
import pyarrow.parquet as pq
import zipfile
import glob
from config import CALL_REPORT_DATA_DIR, CROSSWALK_CSV, CROSSWALK_PARQUET

def build_rssd_cert_crosswalk(zip_folder: str, out_csv: str = "rssd_cert_crosswalk.csv", out_parquet: str = None):
    dfs = []
    zip_files = sorted(glob.glob(os.path.join(zip_folder, "*.zip")))
    for zip_path in zip_files:
        print(f"Processing {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for fname in z.namelist():
                if fname.lower().endswith(".txt") and "readme" not in fname.lower():
                    try:
                        df = pd.read_csv(z.open(fname), sep="\t", encoding="latin1", dtype=str,
                                         usecols=["IDRSSD", "FDIC Certificate Number"], low_memory=False)
                        df = df.rename(columns={"IDRSSD": "ID_RSSD", "FDIC Certificate Number": "CERT"})
                        df = df.dropna().drop_duplicates()
                        dfs.append(df)
                        print(f"    {fname}: {df.shape[0]} rows")
                    except ValueError:
                        print(f"    Skipping {fname}: missing IDRSSD or FDIC Certificate Number")
    if not dfs:
        raise ValueError("No valid files found with IDRSSD and CERT columns.")
    crosswalk = pd.concat(dfs, ignore_index=True).drop_duplicates()
    issues = {}
    rssd_to_cert = crosswalk.groupby("ID_RSSD")["CERT"].nunique()
    multi_rssd = rssd_to_cert[rssd_to_cert > 1]
    if not multi_rssd.empty:
        issues["rssd_conflicts"] = multi_rssd
    cert_to_rssd = crosswalk.groupby("CERT")["ID_RSSD"].nunique()
    multi_cert = cert_to_rssd[cert_to_rssd > 1]
    if not multi_cert.empty:
        issues["cert_conflicts"] = multi_cert
    crosswalk.to_csv(out_csv, index=False)
    print(f"\nFinal crosswalk saved to {out_csv} ({crosswalk.shape[0]} rows)")
    if out_parquet:
        safe_crosswalk = crosswalk.astype({"ID_RSSD": "string", "CERT": "string"})
        safe_crosswalk.to_parquet(out_parquet, index=False, engine="fastparquet")
        print(f"   💾 Also saved Parquet to {out_parquet}")
    if issues:
        print("\nConsistency issues detected:")
        for k, v in issues.items():
            print(f"- {k}: {len(v)} conflicts")
    else:
        print("\nNo consistency issues found — mapping is one-to-one.")
    return crosswalk, issues

# Run with config paths
crosswalk, issues = build_rssd_cert_crosswalk(
    zip_folder=CALL_REPORT_DATA_DIR,
    out_csv=CROSSWALK_CSV,
    out_parquet=CROSSWALK_PARQUET
)
print("\nSample crosswalk:")
print(crosswalk.head())
print("\nConflicts found:", list(issues.keys()))