import os
import zipfile
import pandas as pd
from functools import reduce
from tqdm import tqdm
from config import UBPR_DATA_DIR, UBPR_PANEL_DIR, UBPR_PANEL_FILE

ubpr_vars_pd = [
    "UBPRD486", "UBPRD488", "UBPR7402", "UBPRNC98", "UBPR7408", "UBPRE022",
    "UBPRE021", "UBPRE395", "UBPRE544", "UBPR7414", "UBPRE549", "UBPRE019",
    "UBPRE020", "UBPRE001", "UBPRE002", "UBPRE003", "UBPRE004", "UBPRE005",
    "UBPRPG69", "UBPRE006", "UBPRKW07", "UBPRE007", "UBPRE009", "UBPRE010",
    "UBPRE013", "UBPRK447", "UBPRE014", "UBPRE015", "UBPRE029", "UBPR7316",
    "UBPRE027", "UBPRE028"
]

def merge_ubpr_dict_unique(dfs: dict) -> pd.DataFrame:
    keys = ["ID RSSD", "Reporting Period"]
    renamed_dfs = []
    seen = set()
    for name, df in dfs.items():
        df = df.loc[:, ~df.columns.duplicated()]
        keep_cols = [c for c in df.columns if c in keys or c not in seen]
        seen.update(keep_cols)
        renamed_dfs.append(df[keep_cols])
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), renamed_dfs)
    return merged_df

def process_ubpr_folder(zip_folder: str, out_folder: str, final_file: str = "UBPR_Panel.parquet"):
    os.makedirs(out_folder, exist_ok=True)
    zip_files = sorted([f for f in os.listdir(zip_folder) if f.endswith(".zip")])
    yearly_files = []
    for zf in tqdm(zip_files, desc="Processing years"):
        year = os.path.splitext(os.path.basename(zf))[0].split()[-1]
        dfs = {}
        with zipfile.ZipFile(os.path.join(zip_folder, zf), "r") as z:
            txt_files = [f for f in z.namelist() if f.endswith(".txt") and "readme" not in f.lower()]
            for file in txt_files:
                try:
                    with z.open(file) as f:
                        use_cols = ["ID RSSD", "Reporting Period"] + ubpr_vars_pd
                        df = pd.read_csv(f, sep="\t", encoding="cp1252", usecols=lambda c: c in use_cols,
                                         na_values=[".", "NA", "", " "], low_memory=False)
                        dfs[file] = df
                except Exception as e:
                    print(f"Skipping {file} in {zf} ({e})")
        if dfs:
            merged = merge_ubpr_dict_unique(dfs)
            merged["ID RSSD"] = merged["ID RSSD"].astype("Int64").astype(str)
            merged["Reporting Period"] = merged["Reporting Period"].astype(str)
            for col in merged.columns:
                if col not in ["ID RSSD", "Reporting Period"]:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce")
            out_path = os.path.join(out_folder, f"UBPR_{year}.parquet")
            merged.to_parquet(out_path, index=False)
            yearly_files.append(out_path)
    print("\nConcatenating yearly files into panel...")
    panel_dfs = [pd.read_parquet(f) for f in tqdm(yearly_files, desc="Concatenating")]
    panel = pd.concat(panel_dfs, ignore_index=True)
    panel.to_parquet(os.path.join(out_folder, final_file), index=False)
    print(f"\nFinal panel saved to {os.path.join(out_folder, final_file)} with shape {panel.shape}")

# Run with config paths
process_ubpr_folder(UBPR_DATA_DIR, UBPR_PANEL_DIR, final_file=os.path.basename(UBPR_PANEL_FILE))