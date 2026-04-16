from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import month_end_index, series_from_url


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='.', type=str)
    p.add_argument('--bond-spec', default='data/input/Germany_Bond_Data.xlsx', type=str)
    p.add_argument('--start', default='2000-01-31', type=str)
    p.add_argument('--end', default='2022-07-31', type=str)
    p.add_argument('--fallback-raw-dirs', default='', type=str)
    return p.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        'raw': root / 'data' / 'raw' / 'bond',
        'processed': root / 'data' / 'processed' / 'bond',
        'logs': root / 'outputs' / 'logs',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def run() -> None:
    warnings.filterwarnings('ignore')
    args = parse_args()
    root = Path(args.project_root).resolve()
    spec_path = (root / args.bond_spec).resolve()
    dirs = ensure_dirs(root)

    fallback_dirs = [Path(x).expanduser().resolve() for x in args.fallback_raw_dirs.split(',') if x.strip()]

    spec = pd.read_excel(spec_path)

    yields = {}
    logs = []
    for n in [1, 2, 3, 4, 5]:
        row = spec.loc[spec['Short Name'] == f'y{n}_zc'].iloc[0]
        url = str(row['Direct CSV Link']).strip()
        key = str(row.get('Series Key / Filter', '')).strip()
        s, col, h, p = series_from_url(url, dirs['raw'], key_filter=key, fallback_dirs=fallback_dirs)
        s = month_end_index(s)
        yields[f'y{n}_pct'] = s
        logs.append({
            'Series No': int(row['Series No']),
            'Short Name': row['Short Name'],
            'Source Hash': h,
            'Selected Column': col,
            'First Date': s.index.min().strftime('%Y-%m-%d') if len(s) else '',
            'Last Date': s.index.max().strftime('%Y-%m-%d') if len(s) else '',
            'Raw File': p.name,
        })

    ydf = pd.concat(yields.values(), axis=1)
    ydf.columns = list(yields.keys())
    ydf = ydf.sort_index()

    for n in [1, 2, 3, 4, 5]:
        ydf[f'y{n}_log'] = np.log(1 + ydf[f'y{n}_pct'] / 100.0)
        ydf[f'p{n}_log'] = -float(n) * ydf[f'y{n}_log']

    for n in [2, 3, 4, 5]:
        ydf[f'g{n}_fwd'] = ydf[f'p{n-1}_log'] - ydf[f'p{n}_log']
        ydf[f'r{n}_hpr_1y'] = ydf[f'p{n-1}_log'].shift(-12) - ydf[f'p{n}_log']
        ydf[f'rx{n}_1y'] = ydf[f'r{n}_hpr_1y'] - ydf['y1_log']

    ydf['rx_avg_1y'] = ydf[[f'rx{n}_1y' for n in [2, 3, 4, 5]]].mean(axis=1)

    X = ydf[['y1_log', 'g2_fwd', 'g3_fwd', 'g4_fwd', 'g5_fwd']]
    cp = sm.OLS(ydf['rx_avg_1y'], sm.add_constant(X, has_constant='add'), missing='drop').fit()
    ydf['CP_t'] = cp.predict(sm.add_constant(X, has_constant='add'))

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    ydf_trim = ydf[(ydf.index >= start) & (ydf.index <= end)].copy()

    ydf.to_csv(dirs['processed'] / 'bond_panel_full.csv', index_label='date')
    ydf_trim.to_csv(dirs['processed'] / 'bond_panel_trimmed.csv', index_label='date')
    pd.DataFrame(logs).to_csv(dirs['logs'] / 'bond_download_log.csv', index=False)


if __name__ == '__main__':
    run()
