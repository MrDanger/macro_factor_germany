# Germany Macro-Factor Bond Replication

## Structure
- `scripts/01_prepare_macro_131.py`: 131-series download, cleaning, composite handling, transformations, full/trimmed panel outputs.
- `scripts/02_prepare_bond_data.py`: bond download and construction (`y`, `p`, `g`, `r`, `rx`, `CP_t`), full/trimmed panel outputs.
- `scripts/03_run_replication.py`: factor extraction, regressions, ordered figures and tables.
- `scripts/run_all.py`: full pipeline runner.

## Inputs
- `data/input/Macro Factors for Germany.xlsx`
- `data/input/Germany_Bond_Data.xlsx`

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python scripts/run_all.py --project-root .
```

## Outputs
- Raw downloads: `data/raw/macro_131`, `data/raw/bond`
- Cleaned/full/trimmed panels: `data/processed/macro_131`, `data/processed/bond`
- Replication products: `data/processed/replication`
- Figures (PDF): `outputs/figures`
- Tables (CSV + LaTeX): `outputs/tables`
- Logs: `outputs/logs`

## Optional Flags
- `--start`, `--end`: trim window override (default `2000-01-31` to `2022-07-31`)
- `--fallback-raw-dirs`: comma-separated cache directories
