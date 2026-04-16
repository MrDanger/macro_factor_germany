from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='.', type=str)
    p.add_argument('--macro-transformed', default='data/processed/macro_131/macro_131_transformed_panel_trimmed.csv', type=str)
    p.add_argument('--bond-panel', default='data/processed/bond/bond_panel_trimmed.csv', type=str)
    p.add_argument('--n-factors', default=8, type=int)
    p.add_argument('--nw-lags', default=18, type=int)
    p.add_argument('--initial-oos', default=120, type=int)
    return p.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        'processed': root / 'data' / 'processed' / 'replication',
        'figures': root / 'outputs' / 'figures',
        'tables': root / 'outputs' / 'tables',
        'logs': root / 'outputs' / 'logs',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=1).replace(0, np.nan)
    return (df - mu) / sd


def pca_fhat(Y: pd.DataFrame, r: int) -> tuple[pd.DataFrame, np.ndarray]:
    arr = Y.values
    T = arr.shape[0]
    yy = arr @ arr.T
    U, S, _ = np.linalg.svd(yy, full_matrices=False)
    F = U[:, :r] * np.sqrt(T)
    var_share = S / S.sum()
    cum = np.cumsum(var_share)
    fdf = pd.DataFrame(F, index=Y.index, columns=[f'F{i}' for i in range(1, r + 1)])
    return fdf, cum


def hac_ols(y: pd.Series, X: pd.DataFrame, lags: int):
    Xc = sm.add_constant(X, has_constant='add')
    return sm.OLS(y, Xc, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': lags})


def clark_west(y_true: np.ndarray, f_r: np.ndarray, f_u: np.ndarray) -> float:
    e_r = y_true - f_r
    e_u = y_true - f_u
    cw = e_r**2 - (e_u**2 - (f_r - f_u) ** 2)
    cw = cw[np.isfinite(cw)]
    if len(cw) < 10:
        return np.nan
    return float(cw.mean() / (cw.std(ddof=1) / np.sqrt(len(cw))))


def save_table1(table1: pd.DataFrame, out_tex: Path) -> None:
    with out_tex.open('w', encoding='utf-8') as f:
        f.write('\\begin{table}[htbp]\n\\centering\n')
        f.write('\\caption{Table 1 Analog: Summary Statistics of Estimated Factors}\n')
        f.write('\\begin{tabular}{lrrrr}\\hline\n')
        f.write('Factor & Mean & Std. Dev. & AR(1) & Cumulative $R^2$\\\\\\hline\n')
        for _, r in table1.iterrows():
            f.write(f"{r['Factor']} & {r['Mean']:.3f} & {r['StdDev']:.3f} & {r['AR1']:.3f} & {r['CumR2']:.3f}\\\\\n")
        f.write('\\hline\\end{tabular}\\end{table}\n')


def save_table2(table2: pd.DataFrame, out_tex: Path) -> None:
    with out_tex.open('w', encoding='utf-8') as f:
        f.write('\\begin{table}[p]\n\\centering\n')
        f.write('\\caption{Table 2 Analog: Regression of Monthly Excess Bond Returns on Lagged Factors}\n')
        f.write('\\scriptsize\n')
        f.write('\\begin{tabular}{ccrrrrrrrrrr}\\hline\n')
        f.write('Mat. & Row & $\\hat F_1$ & $\\hat F_1^3$ & $\\hat F_2$ & $\\hat F_3$ & $\\hat F_4$ & $\\hat F_8$ & $CP_t$ & $F5_t$ & $F6_t$ & $\\bar R^2$\\\\\\hline\n')
        for n in [2, 3, 4, 5]:
            d = table2[table2['maturity'] == n].set_index('row')
            for rid in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                rr = d.loc[rid]
                cells = []
                for cn in ['F1', 'F1_cu', 'F2', 'F3', 'F4', 'F8', 'CP_t', 'F5', 'F6']:
                    b = rr[f'b_{cn}']
                    t = rr[f't_{cn}']
                    cells.append('' if pd.isna(b) else f"{b:.3f}\\\\ ({t:.2f})")
                lhs = f"$rx^{{({n})}}_{{t+1}}$" if rid == 'a' else ''
                f.write(' & '.join([lhs, f'({rid})'] + cells + [f"{rr['adj_r2']:.2f}"]) + '\\\\\n')
            f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\vspace{2mm}\\par\\footnotesize Notes: Newey--West HAC t-statistics with lag 18 in parentheses.\n')
        f.write('\\end{table}\n')


def save_table3(table3: pd.DataFrame, out_tex: Path) -> None:
    with out_tex.open('w', encoding='utf-8') as f:
        f.write('\\begin{table}[p]\n\\centering\n')
        f.write('\\caption{Table 3 Analog: Out-of-Sample Predictive Power of Macro Factors}\n')
        f.write('\\small\n')
        f.write('\\begin{tabular}{cccrrrr}\\hline\n')
        f.write('Mat. & Row & Forecast sample & OOS N & $MSE_u/MSE_r$ & Test stat & 95\\% crit.\\\\\\hline\n')
        for _, r in table3.iterrows():
            samp = f"{r['forecast_start']} to {r['forecast_end']}"
            z = '' if pd.isna(r['clark_west_z']) else f"{r['clark_west_z']:.2f}"
            f.write(f"{int(r['maturity'])} & ({r['row']}) & {samp} & {int(r['oos_n'])} & {r['mse_u_over_mse_r']:.3f} & {z} & 1.645\\\\\n")
        f.write('\\hline\\end{tabular}\\end{table}\n')


def run() -> None:
    warnings.filterwarnings('ignore')
    args = parse_args()
    root = Path(args.project_root).resolve()
    dirs = ensure_dirs(root)

    macro = pd.read_csv(root / args.macro_transformed, parse_dates=['date']).set_index('date').sort_index()
    bond = pd.read_csv(root / args.bond_panel, parse_dates=['date']).set_index('date').sort_index()

    # Makedata-style trims in trimmed window: drop first 12, standardize, drop first 48
    m = macro.copy()
    m = m.iloc[12:, :]
    m = m.loc[:, m.notna().mean(axis=0) >= 0.9]
    m = m.dropna(axis=1, how='any')
    m_std = standardize(m)
    m_std = m_std.iloc[48:, :]
    m_std = m_std.dropna(axis=0, how='any')

    fdf, cum_r2 = pca_fhat(m_std, args.n_factors)
    fdf['F1_cu'] = fdf['F1'] ** 3

    merged = bond.join(fdf, how='inner')
    merged.to_csv(dirs['processed'] / 'replication_merged_panel.csv', index_label='date')
    fdf.to_csv(dirs['processed'] / 'factors.csv', index_label='date')

    # Table 1
    t1_rows = []
    for i in range(args.n_factors):
        fi = fdf[f'F{i+1}']
        t1_rows.append({
            'Factor': f'F{i+1}',
            'Mean': float(fi.mean()),
            'StdDev': float(fi.std(ddof=1)),
            'AR1': float(fi.autocorr(lag=1)),
            'CumR2': float(cum_r2[i]),
        })
    table1 = pd.DataFrame(t1_rows)

    # Table 2
    specs = {
        'a': ['CP_t'],
        'b': ['F1', 'F1_cu', 'F2', 'F3', 'F4', 'F8'],
        'c': ['F1', 'F1_cu', 'F2', 'F3', 'F4', 'F8', 'CP_t'],
        'd': ['F1', 'F1_cu', 'F3', 'F4', 'F8'],
        'e': ['F1', 'F1_cu', 'F3', 'F4', 'F8', 'CP_t'],
        'f': ['F5'],
        'g': ['F6'],
        'h': ['CP_t', 'F5'],
    }

    t2_rows = []
    fitted_cache = {}
    for n in [2, 3, 4, 5]:
        ycol = f'rx{n}_1y'
        for rid, xcols in specs.items():
            d = merged[[ycol] + xcols].dropna()
            model = hac_ols(d[ycol], d[xcols], args.nw_lags)
            fitted_cache[(n, rid)] = (model, d.index)
            rec = {'maturity': n, 'row': rid, 'adj_r2': float(model.rsquared_adj), 'nobs': int(model.nobs)}
            for cn in ['F1', 'F1_cu', 'F2', 'F3', 'F4', 'F8', 'CP_t', 'F5', 'F6']:
                rec[f'b_{cn}'] = float(model.params[cn]) if cn in model.params.index else np.nan
                rec[f't_{cn}'] = float(model.tvalues[cn]) if cn in model.tvalues.index else np.nan
            t2_rows.append(rec)
    table2 = pd.DataFrame(t2_rows)

    # Table 3
    t3_rows = []
    for n in [2, 3, 4, 5]:
        ycol = f'rx{n}_1y'
        d_all = merged[[ycol, 'CP_t', 'F1', 'F1_cu', 'F2', 'F3', 'F4', 'F8', 'F5', 'F6']].dropna()
        for rid in ['b', 'c', 'd', 'e', 'f', 'g', 'h']:
            x_r = specs['a']
            x_u = specs[rid]
            y_true = []
            fr = []
            fu = []
            er2 = []
            eu2 = []
            idx = []
            for t in range(args.initial_oos, len(d_all)):
                tr = d_all.iloc[:t]
                te = d_all.iloc[t:t+1]
                mr = sm.OLS(tr[ycol], sm.add_constant(tr[x_r], has_constant='add')).fit()
                mu = sm.OLS(tr[ycol], sm.add_constant(tr[x_u], has_constant='add')).fit()
                pr = float(mr.predict(sm.add_constant(te[x_r], has_constant='add')).iloc[0])
                pu = float(mu.predict(sm.add_constant(te[x_u], has_constant='add')).iloc[0])
                yy = float(te[ycol].iloc[0])
                y_true.append(yy); fr.append(pr); fu.append(pu)
                er2.append((yy - pr) ** 2); eu2.append((yy - pu) ** 2)
                idx.append(te.index[0])
            if len(idx) == 0:
                continue
            t3_rows.append({
                'maturity': n,
                'row': rid,
                'forecast_start': str(min(idx).date()),
                'forecast_end': str(max(idx).date()),
                'oos_n': len(idx),
                'mse_u_over_mse_r': float(np.mean(eu2) / np.mean(er2)),
                'clark_west_z': clark_west(np.array(y_true), np.array(fr), np.array(fu)),
            })
    table3 = pd.DataFrame(t3_rows)

    # Save data tables
    table1.to_csv(dirs['tables'] / 'table1.csv', index=False)
    table2.to_csv(dirs['tables'] / 'table2.csv', index=False)
    table3.to_csv(dirs['tables'] / 'table3.csv', index=False)
    save_table1(table1, dirs['tables'] / 'table1.tex')
    save_table2(table2, dirs['tables'] / 'table2.tex')
    save_table3(table3, dirs['tables'] / 'table3.tex')

    # Figure 1-5: marginal R^2 for selected factors
    for k, fname in [(1, 'figure_1_marginal_r2_F1.pdf'), (2, 'figure_2_marginal_r2_F2.pdf'), (3, 'figure_3_marginal_r2_F3.pdf'), (4, 'figure_4_marginal_r2_F4.pdf'), (8, 'figure_5_marginal_r2_F8.pdf')]:
        ff = fdf[f'F{k}'].reindex(m_std.index)
        r2 = []
        for c in m_std.columns:
            d = pd.concat([m_std[c], ff], axis=1).dropna()
            if len(d) < 20:
                r2.append(np.nan)
            else:
                mod = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1], has_constant='add')).fit()
                r2.append(mod.rsquared)
        x = np.arange(1, len(r2) + 1)
        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.bar(x, r2, width=0.9, color='#4e79a7')
        ax.set_title(f'Marginal R-squares for F{k}')
        ax.set_xlabel('Series Index')
        ax.set_ylabel('R-squared')
        ax.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(dirs['figures'] / fname)
        plt.close(fig)

    # Figure 6A/B
    ip_col = 'S006_IP_total' if 'S006_IP_total' in m_std.columns else m_std.columns[0]
    ma = pd.concat({
        'F1': ((fdf['F1'] - fdf['F1'].mean()) / fdf['F1'].std(ddof=1)),
        'F5': ((fdf['F5'] - fdf['F5'].mean()) / fdf['F5'].std(ddof=1)),
        'IPg': ((m_std[ip_col] - m_std[ip_col].mean()) / m_std[ip_col].std(ddof=1)),
    }, axis=1).rolling(12, min_periods=12).mean().dropna()

    fig, axs = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)
    axs[0].plot(ma.index, ma['F1'], lw=1.5, label='F1 (12M MA, z)')
    axs[0].plot(ma.index, ma['IPg'], lw=1.3, label='IP growth proxy (12M MA, z)')
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.25)
    axs[1].plot(ma.index, ma['F5'], lw=1.5, label='F5 (12M MA, z)')
    axs[1].plot(ma.index, ma['IPg'], lw=1.3, label='IP growth proxy (12M MA, z)')
    axs[1].legend(frameon=False)
    axs[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(dirs['figures'] / 'figure_6_panels_A_B.pdf')
    plt.close(fig)

    # Figure 7-10 analogs from model-implied premium proxies
    mod5a, idxa = fitted_cache[(5, 'a')]
    mod5c, idxc = fitted_cache[(5, 'c')]
    d5a = merged.loc[idxa, ['CP_t']]
    d5c = merged.loc[idxc, specs['c']]
    rp = pd.DataFrame(index=merged.index)
    rp['ret_with'] = np.nan
    rp['ret_without'] = np.nan
    rp.loc[idxc, 'ret_with'] = mod5c.predict(sm.add_constant(d5c, has_constant='add')) * 100
    rp.loc[idxa, 'ret_without'] = mod5a.predict(sm.add_constant(d5a, has_constant='add')) * 100
    rp['yield_with'] = rp['ret_with'] / 5
    rp['yield_without'] = rp['ret_without'] / 5
    rp_ma = rp.rolling(12, min_periods=12).mean()

    fig, axs = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)
    axs[0].plot(rp_ma.index, rp_ma['yield_with'], lw=1.5, label='Yield premium proxy (with factors)')
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.25)
    axs[1].plot(ma.index, ma['IPg'], lw=1.3, label='IP growth proxy')
    axs[1].legend(frameon=False)
    axs[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(dirs['figures'] / 'figure_7_panels_A_B.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(rp_ma.index, rp_ma['yield_with'], lw=1.5, label='With factors')
    ax.plot(rp_ma.index, rp_ma['yield_without'], lw=1.4, label='Without factors')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(dirs['figures'] / 'figure_8_yield_premium_with_vs_without_factors.pdf')
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)
    axs[0].plot(rp_ma.index, rp_ma['ret_with'], lw=1.5, label='Return premium with factors')
    axs[0].plot(ma.index, ma['IPg'], lw=1.3, label='IP growth proxy')
    axs[0].legend(frameon=False)
    axs[0].grid(alpha=0.25)
    axs[1].plot(rp_ma.index, rp_ma['ret_with'], lw=1.5, label='With factors')
    axs[1].plot(rp_ma.index, rp_ma['ret_without'], lw=1.3, label='Without factors')
    axs[1].legend(frameon=False)
    axs[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(dirs['figures'] / 'figure_9_panels_A_B.pdf')
    plt.close(fig)

    dec = pd.DataFrame(index=merged.index)
    dec['y5'] = merged['y5_pct']
    dec['term_premium_proxy'] = rp['yield_with']
    dec['expectations_proxy'] = dec['y5'] - dec['term_premium_proxy']
    dec = dec.dropna()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dec.index, dec['y5'], lw=1.8, label='5-year yield')
    ax.plot(dec.index, dec['term_premium_proxy'], lw=1.4, label='Term premium proxy')
    ax.plot(dec.index, dec['expectations_proxy'], lw=1.4, label='Expectations proxy')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(dirs['figures'] / 'figure_10_yield_decomposition_with_factors.pdf')
    plt.close(fig)

    dec.to_csv(dirs['processed'] / 'figure10_decomposition.csv', index_label='date')


if __name__ == '__main__':
    run()
