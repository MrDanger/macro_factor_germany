from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import hashlib
import io
import re
import subprocess

import numpy as np
import pandas as pd
import requests


SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Mozilla/5.0'})


def month_end_index(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.to_datetime(out.index).to_period('M').to_timestamp(how='end').normalize()
    out = out.groupby(out.index).last().sort_index()
    return out


def infer_frequency(index: pd.DatetimeIndex) -> str:
    d = pd.to_datetime(index).sort_values()
    if len(d) < 3:
        return 'Unknown'
    dd = pd.Series(d).diff().dropna().dt.days
    med = dd.median()
    if 25 < med < 35:
        return 'Monthly'
    if 70 < med < 100:
        return 'Quarterly'
    return 'Irregular'


def to_monthly(series: pd.Series, tran_label: str = '') -> tuple[pd.Series, str]:
    s = series.dropna().sort_index()
    if s.empty:
        return s, 'empty'
    freq = infer_frequency(pd.DatetimeIndex(s.index))
    if freq == 'Monthly':
        return month_end_index(s), 'none'
    if freq != 'Quarterly':
        return month_end_index(s), 'irregular_as_points'

    q = s.copy()
    q.index = q.index.to_period('Q').asfreq('M', 'end').to_timestamp(how='end').normalize()
    q = q[~q.index.duplicated(keep='last')]
    mi = pd.date_range(start=q.index.min(), end=q.index.max(), freq='ME')
    out = q.reindex(mi)
    if tran_label in ['ln', 'Δln', 'Δ2ln'] and (q > 0).all():
        out = np.exp(np.log(out).interpolate(method='time'))
        return out, 'quarterly_log_linear_interp'
    out = out.interpolate(method='time')
    return out, 'quarterly_linear_interp'


def apply_transform(series: pd.Series, tran_label: str) -> pd.Series:
    x = series.astype(float).copy()
    small = 1e-6
    if tran_label == 'lv':
        return x
    if tran_label == 'Δlv':
        return x.diff(1)
    if tran_label == 'ln':
        y = x.copy()
        y[x < small] = np.nan
        return np.log(y)
    if tran_label == 'Δln':
        y = x.copy()
        y[x < small] = np.nan
        return np.log(y).diff(1)
    if tran_label == 'Δ2ln':
        y = x.copy()
        y[x < small] = np.nan
        return np.log(y).diff(1).diff(1)
    return x * np.nan


def split_links(x: object) -> list[str]:
    if pd.isna(x):
        return []
    t = str(x).replace('\xa0', ' ').strip()
    if not t:
        return []
    return [u.strip() for u in re.split(r'[\n\r;]+', t) if u.strip()]


def cache_fetch(url: str, raw_dir: Path, fallback_dirs: list[Path] | None = None) -> tuple[str, str, Path]:
    h = hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]
    p = raw_dir / f'{h}.csv'
    if p.exists():
        return p.read_text(encoding='utf-8', errors='ignore'), h, p
    for d in (fallback_dirs or []):
        fp = d / f'{h}.csv'
        if fp.exists():
            txt = fp.read_text(encoding='utf-8', errors='ignore')
            p.write_text(txt, encoding='utf-8')
            return txt, h, p
    try:
        r = subprocess.run(['curl', '-L', '--silent', '--show-error', '--max-time', '60', url], capture_output=True, text=True, check=False)
        if r.returncode == 0 and r.stdout:
            txt = r.stdout
            p.write_text(txt, encoding='utf-8')
            return txt, h, p
    except Exception:
        pass
    txt = SESSION.get(url, timeout=90).text
    p.write_text(txt, encoding='utf-8')
    return txt, h, p


def parse_fred(text: str) -> tuple[pd.Series, str]:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce'), 'value': pd.to_numeric(df[val_col], errors='coerce')}).dropna(subset=['date'])
    s = pd.Series(out['value'].values, index=out['date']).sort_index()
    return s, val_col


def parse_bundesbank(text: str) -> tuple[pd.Series, str]:
    vals = []
    for ln in text.splitlines():
        m = re.match(r'^\s*(\d{4}-\d{2});\s*([-+]?[0-9]+(?:,[0-9]+)?)\s*;?', ln)
        if m:
            d = pd.to_datetime(m.group(1) + '-01').to_period('M').to_timestamp('M')
            v = float(m.group(2).replace(',', '.'))
            vals.append((d, v))
    if not vals:
        for ln in text.splitlines():
            m = re.match(r'^(\d{4}-\d{2}),(.*?)(,.*)?$', ln.strip('\ufeff'))
            if m:
                d = pd.to_datetime(m.group(1) + '-01').to_period('M').to_timestamp('M')
                v = float(m.group(2).replace('.', '').replace(',', '.'))
                vals.append((d, v))
    if not vals:
        raise ValueError('bundesbank_parse_failed')
    s = pd.Series(dict(vals)).sort_index()
    return s, 'value'


def parse_dbnomics(text: str) -> tuple[pd.Series, str]:
    df = pd.read_csv(io.StringIO(text))
    val_col = [c for c in df.columns if c != 'period'][0]
    date = pd.PeriodIndex(df['period'].astype(str), freq='Q').to_timestamp(how='end').normalize()
    s = pd.Series(pd.to_numeric(df[val_col], errors='coerce').values, index=date).sort_index()
    return s, val_col


def parse_eurostat(text: str, key_filter: str = '') -> tuple[pd.Series, str]:
    df = pd.read_csv(io.StringIO(text))
    dff = df.copy()
    dim_cols = [c for c in dff.columns if c not in ['STRUCTURE', 'STRUCTURE_ID', 'TIME_PERIOD', 'OBS_VALUE', 'OBS_FLAG', 'CONF_STATUS']]
    if isinstance(key_filter, str) and ':' in key_filter:
        tokens = [p.strip() for p in key_filter.split(':')[1:] if p.strip()]
        for tok in tokens:
            m = pd.Series(False, index=dff.index)
            for c in dim_cols:
                m = m | (dff[c].astype(str).str.upper() == tok.upper())
            dff = dff[m]
    if dff.empty:
        dff = df.copy()
    if dim_cols:
        grp = dff.groupby(dim_cols, dropna=False)['OBS_VALUE'].count().reset_index(name='n').sort_values('n', ascending=False)
        if len(grp):
            best = grp.iloc[0]
            m = pd.Series(True, index=dff.index)
            for c in dim_cols:
                m &= (dff[c].astype(str) == str(best[c]))
            dff = dff[m]
    date = pd.to_datetime(dff['TIME_PERIOD'], errors='coerce')
    val = pd.to_numeric(dff['OBS_VALUE'], errors='coerce')
    s = pd.Series(val.values, index=date).dropna().sort_index()
    return s, 'OBS_VALUE'


def parse_generic_csv(text: str) -> tuple[pd.Series, str]:
    df = pd.read_csv(io.StringIO(text))
    cols = list(df.columns)
    date_col = cols[0]
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ['date', 'time', 'period', 'observation']):
            date_col = c
            break
    num_cols = [c for c in cols if c != date_col]
    val_col = num_cols[0] if num_cols else cols[-1]
    date = pd.to_datetime(df[date_col], errors='coerce')
    val = pd.to_numeric(df[val_col], errors='coerce')
    s = pd.Series(val.values, index=date).dropna().sort_index()
    return s, val_col


def parse_by_url(url: str, text: str, key_filter: str = '') -> tuple[pd.Series, str]:
    host = urlparse(url).netloc.lower()
    if 'fred.stlouisfed.org' in host:
        return parse_fred(text)
    if 'bundesbank.de' in host:
        return parse_bundesbank(text)
    if 'api.db.nomics.world' in host:
        return parse_dbnomics(text)
    if 'ec.europa.eu' in host:
        return parse_eurostat(text, key_filter=key_filter)
    return parse_generic_csv(text)


def series_from_url(url: str, raw_dir: Path, key_filter: str = '', fallback_dirs: list[Path] | None = None) -> tuple[pd.Series, str, str, Path]:
    txt, h, path = cache_fetch(url, raw_dir=raw_dir, fallback_dirs=fallback_dirs)
    s, col = parse_by_url(url, txt, key_filter=key_filter)
    return s, col, h, path
