# PD_FFIEC: Bank Default Probability Model

This project estimates the probability of default (PD) for U.S. banks using FFIEC UBPR and Call Report data, showcasing my data science and financial modeling skills for future employers.

## Overview

The model predicts bank defaults using financial ratios aligned with CAMELS ratings, implemented with Python, Pandas, LightGBM, and Jupyter notebooks.

- **Data**: FFIEC UBPR, Call Report, FDIC Failed Bank List.
- **Model**: Logistic Regression (Ridge) and LightGBM for binary classification.
- **Outputs**: Precision-recall curves, feature importances in `reports/`.

## Key Variables

Selected from FFIEC UBPR User's Guide, these ratios assess capital adequacy, asset quality, earnings, liquidity, and management/growth.

| Category              | UBPR Code   | Description                                      | Narrative/Formula                                                                 | Call Report Codes |
|-----------------------|-------------|--------------------------------------------------|-----------------------------------------------------------------------------------|-------------------|
| Capital & Reserves    | UBPRD486   | Tier One Leverage Capital                       | `IF(uc:UBPR9999[P0] > '2001-01-01', (uc:UBPR7204[P0]*100), null)` | RCFD7204, RCON7204 |
| Capital & Reserves    | UBPRD488   | Total Risk-Based Capital                        | `IF(uc:UBPR9999[P0] > '2001-01-01', (uc:UBPR7205[P0]*100), null)` | RCFD7205, RCON7205 |
| Capital & Reserves    | UBPR7402   | Cash Dividends to Net Income                    | `IF(cc:RIAD4340[P0] <> 0, PCTOF(uc:UBPRE625[P0], cc:RIAD4340[P0]), NULL)` | RIAD4340, RIAD4460, RIAD4470 |
| Capital & Reserves    | UBPRNC98   | Non-Current Loans & OREO / Tier 1 + ACL         | `PCTOF(uc:UBPRD261[P0], uc:UBPR3792[P0])` | - |
| Capital & Reserves    | UBPR7408   | Tier One Capital Growth Rate                    | `PCTOF(uc:UBPRD347[P0], uc:UBPRD349[P0])` | - |
| Asset Quality         | UBPRE022   | ACL on LN&LS to LN&LS HFI                      | `PCTOF(uc:UBPR3123[P0], uc:UBPRB528[P0])` | RCFD3123, RCON3123, RCFDB528, RCONB528 |
| Asset Quality         | UBPRE021   | ACL to Net Loss (X)                            | `IF(uc:UBPR1795[P0] > 0, PCT(uc:UBPR3123[P0], uc:UBPRD236[P0]), NULL)` | RCFD3123, RCON3123 |
| Asset Quality         | UBPRE395   | ACL to Nonaccrual LN&LS (X)                    | `PCT(uc:UBPR3123[P0], uc:UBPRD669[P0])` | RCFD3123, RCON3123 |
| Asset Quality         | UBPRE544   | LN&LS 30-89 Days Past Due %                    | `PCTOF(uc:UBPRD668[P0], uc:UBPRE131[P0])` | - |
| Asset Quality         | UBPR7414   | Noncurrent Loans to Gross Loans                | `PCTOF(uc:UBPR1400[P0], uc:UBPRE131[P0])` | - |
| Asset Quality         | UBPRE549   | Non-Current LNS+OREO to LNS+OREO               | `PCTOF(uc:UBPRD261[P0], uc:UBPRD270[P0])` | - |
| Asset Quality         | UBPRE019   | Net Loss % of Avg Loans                        | `PCTOFANN(uc:UBPR1795[P0], uc:UBPRE386[P0])` | RIAD4605, RIAD4635, RIADC079, RIAD5523 |
| Asset Quality         | UBPRE020   | Earnings Coverage of Net Losses (X)            | `IF(uc:UBPR1795[P0] > 0, PCT(uc:UBPRD468[P0], uc:UBPR1795[P0]), NULL)` | - |
| Earnings              | UBPRE001   | Interest Income (TE) % of Avg Assets           | `PCTOFANN(uc:UBPR4107[P0], uc:UBPRD659[P0])` | RIAD4107, UBPRD659 |
| Earnings              | UBPRE002   | Interest Expense % of Avg Assets               | `PCTOFANN(cc:RIAD4073[P0], uc:UBPRD659[P0])` | RIAD4073 |
| Earnings              | UBPRE003   | Net Interest Income (TE) % of Avg Assets       | `PCTOFANN(uc:UBPR4074[P0], uc:UBPRD659[P0])` | RIAD4074, UBPRD659 |
| Earnings              | UBPRE004   | Noninterest Income % of Avg Assets             | `PCTOFANN(cc:RIAD4079[P0], uc:UBPRD659[P0])` | RIAD4079 |
| Earnings              | UBPRE005   | Non-Interest Expense % of Avg Assets           | `PCTOFANN(uc:UBPRE037[P0], uc:UBPRD659[P0])` | RIAD4135, RIAD4217, RIAD4092, RIADC216, RIADC232 |
| Earnings              | UBPRPG69   | Pre-Provision Net Revenue % Avg Assets         | `PCTOFANN(uc:UBPRPG64[P0], uc:UBPRD659[P0])` | - |
| Earnings              | UBPRE006   | Provision for Loan Losses % Avg Assets         | `PCTOFANN(uc:UBPRD483[P0], uc:UBPRD659[P0])` | RIAD4230 |
| Earnings              | UBPRKW07   | Provision for Other Assets % Avg Assets        | `PCTOFANN(uc:UBPRKW06[P0], uc:UBPRD659[P0])` | RIADJH90, RIADJH96, RIADJJ02, RIADJJ33 |
| Earnings              | UBPRE007   | Pretax Operating Income (TE) % Avg Assets      | `PCTOFANN(uc:UBPRE038[P0], uc:UBPRD659[P0])` | RIAD4301 |
| Earnings              | UBPRE009   | Pretax Net Operating Income % Avg Assets       | `PCTOFANN(uc:UBPRE039[P0], uc:UBPRD659[P0])` | RIAD4301 |
| Earnings              | UBPRE010   | Net Operating Income % Avg Assets              | `IF(uc:UBPRD659[P0] <> 0, PCTOFANN(cc:RIAD4300[P0], uc:UBPRD659[P0]), NULL)` | RIAD4300 |
| Earnings              | UBPRE013   | Net Income % Avg Assets                        | `IF(uc:UBPRD659[P0] <> 0, PCTOFANN(cc:RIAD4340[P0], uc:UBPRD659[P0]), NULL)` | RIAD4340 |
| Liquidity & Funding   | UBPRK447   | Net Non Core Funding Dependence                | `PCTOF(uc:UBPRK446[P0], uc:UBPRD584[P0])` | RCONJ474, RCON2604, RCONHK05, RCONJ472, RCON2343, RCON2344 |
| Liquidity & Funding   | UBPRE014   | Avg Earning Assets % Avg Assets                | `PCTOF(uc:UBPRD362[P0], uc:UBPRD659[P0])` | RCFD3360, RCON3360, RCFD3484, RCON3484 |
| Liquidity & Funding   | UBPRE015   | Avg Interest-Bearing Funds % Avg Assets        | `PCTOF(uc:UBPRD435[P0], uc:UBPRD659[P0])` | - |
| Liquidity & Funding   | UBPRE029   | Short Term Non Core Funding Growth Rate        | `PCTOF(uc:UBPRD545[P0], uc:UBPRD547[P0])` | RCONHK06, RCONK222, RCONA245, RCONHK17, RCONHK16 |
| Management & Growth   | UBPR7316   | Total Assets - Annual Change                   | `PCTOF(uc:UBPRD087[P0], uc:UBPRD088[P0])` | RCFD2170, RCON2170 |
| Management & Growth   | UBPRE027   | Net Loans and Leases Growth Rate               | `PCTOF(uc:UBPRD250[P0], uc:UBPRD251[P0])` | - |
| Management & Growth   | UBPRE028   | Short Term Investments Growth Rate             | `PCTOF(uc:UBPRD430[P0], uc:UBPRD431[P0])` | RCFD0071, RCON0071 |

## Models

The `models.ipynb` notebook implements two models to predict bank defaults using the latest observation per bank from `analysis_data.parquet`, reducing imbalance and intra-bank correlation:

- **Logistic Regression (Ridge)**:
  - Uses `LogisticRegression(penalty='l2', solver='saga', max_iter=5000, class_weight='balanced')`.
  - Features: 32 financial ratios (e.g., `Tier1_Leverage_Ratio`, `Noncurrent_Loans_to_Gross_Loans`).
  - Data: Median imputation, StandardScaler, 70/30 train-test split (stratified).
  - Performance (threshold=0.82):
    - Precision: 0.99 (non-failure), 0.86 (failure).
    - Recall: 0.99 (non-failure), 0.89 (failure).
    - F1-Score: 0.99 (non-failure), 0.87 (failure).
    - Accuracy: 0.99, ROC-AUC: 0.9796.
    - Support: 2923 non-failures, 151 failures.

- **LightGBM**:
  - Uses `LGBMClassifier(n_estimators=100, early_stopping_rounds=10)`.
  - Same features and preprocessing.
  - Data: Median imputation, StandardScaler, 70/30 train-test split (stratified).
  - Performance (threshold=0.5):
   - Precision: 0.99 (non-failure), 0.89 (failure).
   - Recall: 0.99 (non-failure), 0.89 (failure).
   - F1-Score: 0.99 (non-failure), 0.89 (failure).
   - Accuracy: 0.99, ROC-AUC: 0.9685.
   - Support: 2923 non-failures, 151 failures.

  - Top Predictors (by importance):
    - `Noncurrent_Loans_OREO_to_Tier1_ACL` (230).
    - `Total_Risk_Based_Capital_Ratio` (225).
    - `Loans_30_89_Days_Past_Due_Ratio` (194).
    - `Tier1_Leverage_Ratio` (190).
    - Others: Interest expense, funding, and growth ratios.
  - Strong performance, emphasizing asset quality and capital ratios.

## Folder Structure

- `call_report_processing.py`: Creates RSSD-to-CERT crosswalk from Call Report data.
- `config.py`: Defines file paths.
- `ubpr_processing.py`: Processes UBPR data into `UBPR_Panel.parquet`.
- `data_wrangling.ipynb`: Cleans and merges data, saving to `analysis_data.parquet`.
- `eda.ipynb`: Performs EDA with visualizations.
- `models.ipynb`: Trains Logistic Regression and LightGBM models, generates outputs.
- `data/`: Raw and processed data (excluded via `.gitignore`).
- `reports/`: Model outputs (excluded via `.gitignore`).
- `.gitignore`: Excludes `data/`, `reports/`, `notebooks/`, `scripts/`, etc.

## Setup

1. Clone: `git clone https://github.com/rbenmac/PD_FFIEC.git`
2. Set up env: `python -m venv .venv; source .venv/bin/activate`
3. Install: `pip install pandas numpy pyarrow fastparquet tqdm matplotlib seaborn scikit-learn lightgbm`
4. Add data to `data/raw/`.
5. Run: `python ubpr_processing.py; python all_report_processing.py`
6. Open Jupyter: `jupyter notebook`
   - Run `data_wrangling.ipynb`, `eda.ipynb`, `models.ipynb`.

## Skills

- **Data Engineering**: Handled large FFIEC datasets with Pandas, PyArrow.
- **Machine Learning**: Built and evaluated Logistic Regression and LightGBM models.
- **EDA**: Created visualizations with Matplotlib, Seaborn.
- **Software Development**: Modular scripts and notebooks.
- **Git**: Managed version control.

## Contact

Contact me at [LinkedIn](www.linkedin.com/in/robson-machado-j272913107) to discuss my data science and finance expertise!


