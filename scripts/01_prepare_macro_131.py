from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from common import (
    apply_transform,
    infer_frequency,
    month_end_index,
    series_from_url,
    split_links,
    to_monthly,
)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - mu) / sd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='.', type=str)
    p.add_argument('--macro-spec', default='data/input/Macro Factors for Germany.xlsx', type=str)
    p.add_argument('--start', default='2000-01-31', type=str)
    p.add_argument('--end', default='2022-07-31', type=str)
    p.add_argument('--fallback-raw-dirs', default='', type=str)
    return p.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        'raw': root / 'data' / 'raw' / 'macro_131',
        'series': root / 'data' / 'processed' / 'macro_131' / 'series',
        'panel': root / 'data' / 'processed' / 'macro_131',
        'logs': root / 'outputs' / 'logs',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('_')


def run() -> None:
    warnings.filterwarnings('ignore')
    args = parse_args()
    root = Path(args.project_root).resolve()
    spec_path = (root / args.macro_spec).resolve()
    dirs = ensure_dirs(root)

    fallback_dirs = [Path(x).expanduser().resolve() for x in args.fallback_raw_dirs.split(',') if x.strip()]

    df = pd.read_excel(spec_path)
    keep_status = {'close', 'proxy', 'composite'}

    row_logs = []
    raw_panel = {}
    trans_panel = {}

    policy_rate_proxy = None

    for _, row in df.sort_values('Series No').iterrows():
        rno = int(row['Series No'])
        short = str(row['Short Name']).strip()
        tran = str(row['Tran']).strip()
        status = str(row['Status']).strip().lower()
        key_filter = str(row.get('Series Key / Filter', '')).strip()

        base_log = {
            'Series No': rno,
            'Short Name': short,
            'Status': row['Status'],
            'Process Status': 'SKIPPED',
            'Detected Frequency': '',
            'Raw First Date': '',
            'Raw Last Date': '',
            'Quarterly->Monthly Method': '',
            'Selected Numeric Column': '',
            'Raw File': '',
            'Clean File': '',
            'Composite Formula': '',
            'Notes': '',
        }

        if status not in keep_status:
            row_logs.append(base_log)
            continue

        try:
            direct_url = str(row.get('Direct CSV Link', '')).strip()
            other_links = split_links(row.get('Other Required Direct CSV Links', ''))

            series_monthly = None
            selected_col = ''
            raw_files = []
            comp_formula = ''
            note = ''

            if status == 'composite':
                if rno == 2:
                    raise RuntimeError('unresolved_transfer_component')

                if rno == 22:
                    s1, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    ulink = [u for u in other_links if 'LMUNRLTTDEM647S' in u]
                    s2, c2, _, p2 = series_from_url((ulink[0] if ulink else 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=LMUNRLTTDEM647S'), dirs['raw'], fallback_dirs=fallback_dirs)
                    series_monthly = month_end_index(s1) / month_end_index(s2)
                    selected_col = f'{c1}/{c2}'
                    comp_formula = 'vacancy_ratio = vacancies / unemployment'
                    raw_files = [p1.name, p2.name]

                elif rno == 73:
                    s_m2, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    hicp_url = other_links[0] if other_links else 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CP0000DEM086NEST'
                    s_h, c2, _, p2 = series_from_url(hicp_url, dirs['raw'], fallback_dirs=fallback_dirs)
                    series_monthly = month_end_index(s_m2) / month_end_index(s_h)
                    selected_col = f'{c1}/{c2}'
                    comp_formula = 'real_M2 = nominal_M2 / HICP'
                    raw_files = [p1.name, p2.name]

                elif rno in [94, 95, 96, 97, 98, 99, 100, 101]:
                    try:
                        s_main, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    except Exception:
                        if rno in [96, 97]:
                            s_main, c1, _, p1 = series_from_url('https://fred.stlouisfed.org/graph/fredgraph.csv?id=INTGSTDEM193N', dirs['raw'], fallback_dirs=fallback_dirs)
                        elif rno == 98:
                            s_main, c1, _, p1 = series_from_url('https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N', dirs['raw'], fallback_dirs=fallback_dirs)
                        else:
                            raise
                    if policy_rate_proxy is None:
                        s_pol, _, _, _ = series_from_url('https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBDFR', dirs['raw'], fallback_dirs=fallback_dirs)
                        policy_rate_proxy = s_pol.resample('ME').mean()
                    series_monthly = month_end_index(s_main) - policy_rate_proxy.reindex(month_end_index(s_main).index)
                    selected_col = f'{c1}-ECBDFR'
                    comp_formula = 'spread = main_rate - monthly_avg(ECBDFR)'
                    raw_files = [p1.name]

                elif rno == 107:
                    s1, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    s2, c2, _, p2 = series_from_url(other_links[0], dirs['raw'], fallback_dirs=fallback_dirs)
                    b = pd.concat([month_end_index(s1).rename('a'), month_end_index(s2).rename('b')], axis=1)
                    series_monthly = np.sqrt(b['a'] * b['b'])
                    selected_col = f'geo_mean({c1},{c2})'
                    comp_formula = 'finished_goods_proxy = sqrt(PPI_consumer_goods * PPI_investment_goods)'
                    raw_files = [p1.name, p2.name]

                elif rno == 128:
                    s1, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    s2, c2, _, p2 = series_from_url(other_links[0], dirs['raw'], fallback_dirs=fallback_dirs)
                    q1 = s1.copy(); q1.index = q1.index.to_period('Q').to_timestamp('Q')
                    q2 = s2.copy(); q2.index = q2.index.to_period('Q').to_timestamp('Q')
                    q = pd.concat([q1.rename('be'), q2.rename('f')], axis=1).mean(axis=1)
                    series_monthly, note = to_monthly(q, tran_label=tran)
                    selected_col = f'avg({c1},{c2})'
                    comp_formula = 'goods_earnings_proxy_q = (LCI_B-E + LCI_F)/2'
                    raw_files = [p1.name, p2.name]

                elif rno == 4:
                    s, c, _, p = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    series_monthly = month_end_index(s)
                    selected_col = c
                    comp_formula = 'fallback_proxy = retail turnover direct series'
                    raw_files = [p.name]

                elif rno in [8, 9]:
                    s, c, _, p = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    series_monthly = month_end_index(s)
                    selected_col = c
                    comp_formula = 'fallback_proxy = direct FRED series'
                    raw_files = [p.name]

                elif rno == 64:
                    s1, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    s2, c2, _, p2 = series_from_url(other_links[0], dirs['raw'], fallback_dirs=fallback_dirs)
                    s3, c3, _, p3 = series_from_url(other_links[1], dirs['raw'], fallback_dirs=fallback_dirs)
                    m = pd.concat([month_end_index(s1).rename('i'), month_end_index(s2).rename('d'), month_end_index(s3).rename('n')], axis=1)
                    g = np.log(m).diff().mean(axis=1)
                    idx = np.exp(g.fillna(0).cumsum())
                    series_monthly = 100 * idx / idx.dropna().iloc[0] if idx.dropna().size else idx
                    selected_col = f'avg_growth({c1},{c2},{c3})'
                    comp_formula = 'orders_proxy from averaged growth'
                    raw_files = [p1.name, p2.name, p3.name]

                elif rno == 69:
                    s1, c1, _, p1 = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    s2, c2, _, p2 = series_from_url(other_links[0], dirs['raw'], fallback_dirs=fallback_dirs)
                    s3, c3, _, p3 = series_from_url(other_links[1], dirs['raw'], fallback_dirs=fallback_dirs)
                    s4, c4, _, p4 = series_from_url(other_links[2], dirs['raw'], fallback_dirs=fallback_dirs)
                    m = pd.concat([
                        month_end_index(s1).rename('inv_mfg'),
                        month_end_index(s2).rename('inv_ret'),
                        month_end_index(s3).rename('sal_mfg'),
                        month_end_index(s4).rename('sal_ret'),
                    ], axis=1)
                    series_monthly = (zscore(m['inv_mfg']) + zscore(m['inv_ret'])) / 2 - (zscore(m['sal_mfg']) + zscore(m['sal_ret'])) / 2
                    selected_col = f'z({c1},{c2})-z({c3},{c4})'
                    comp_formula = 'inventory_sales_spread'
                    raw_files = [p1.name, p2.name, p3.name, p4.name]

                else:
                    s, c, _, p = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                    series_monthly = month_end_index(s)
                    selected_col = c
                    comp_formula = 'fallback_direct'
                    raw_files = [p.name]

            else:
                try:
                    s, c, _, p = series_from_url(direct_url, dirs['raw'], key_filter=key_filter, fallback_dirs=fallback_dirs)
                except Exception:
                    if rno in [88, 89]:
                        s, c, _, p = series_from_url('https://fred.stlouisfed.org/graph/fredgraph.csv?id=INTGSTDEM193N', dirs['raw'], fallback_dirs=fallback_dirs)
                    elif rno == 90:
                        s, c, _, p = series_from_url('https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01DEM156N', dirs['raw'], fallback_dirs=fallback_dirs)
                    else:
                        raise
                series_monthly, note = to_monthly(month_end_index(s), tran_label=tran)
                selected_col = c
                raw_files = [p.name]

            series_monthly = series_monthly.sort_index()
            cname = f'S{rno:03d}_{sanitize(short)}'
            series_file = dirs['series'] / f'{cname}.csv'
            pd.DataFrame({'date': series_monthly.index, 'value': series_monthly.values}).to_csv(series_file, index=False)

            transformed = apply_transform(series_monthly, tran)
            raw_panel[cname] = series_monthly
            trans_panel[cname] = transformed

            freq = infer_frequency(pd.DatetimeIndex(series_monthly.dropna().index))
            first = series_monthly.dropna().index.min()
            last = series_monthly.dropna().index.max()

            row_logs.append({
                **base_log,
                'Process Status': 'OK',
                'Detected Frequency': freq,
                'Raw First Date': '' if pd.isna(first) else first.strftime('%Y-%m-%d'),
                'Raw Last Date': '' if pd.isna(last) else last.strftime('%Y-%m-%d'),
                'Quarterly->Monthly Method': note,
                'Selected Numeric Column': selected_col,
                'Raw File': '|'.join(raw_files),
                'Clean File': str(series_file.relative_to(root)),
                'Composite Formula': comp_formula,
                'Notes': '',
            })

        except Exception as e:
            row_logs.append({
                **base_log,
                'Process Status': 'FAILED',
                'Notes': str(e),
            })

    raw_df = pd.DataFrame(raw_panel).sort_index()
    trans_df = pd.DataFrame(trans_panel).sort_index()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    raw_trim = raw_df[(raw_df.index >= start) & (raw_df.index <= end)].copy()
    trans_trim = trans_df[(trans_df.index >= start) & (trans_df.index <= end)].copy()

    raw_df.to_csv(dirs['panel'] / 'macro_131_raw_panel_full.csv', index_label='date')
    raw_trim.to_csv(dirs['panel'] / 'macro_131_raw_panel_trimmed.csv', index_label='date')
    trans_df.to_csv(dirs['panel'] / 'macro_131_transformed_panel_full.csv', index_label='date')
    trans_trim.to_csv(dirs['panel'] / 'macro_131_transformed_panel_trimmed.csv', index_label='date')

    pd.DataFrame(row_logs).sort_values('Series No').to_csv(dirs['logs'] / 'macro_131_row_log.csv', index=False)


if __name__ == '__main__':
    run()
