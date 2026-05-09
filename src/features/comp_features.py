import numpy as np
import pandas as pd

FLOOR_PREMIUM = 4_000  # $/floor level (HDB market rule of thumb: $3–5K)


def _to_month_int(ym_series):
    """Convert 'YYYY-MM' string series to integer month index (YYYY*12 + MM)."""
    parts = ym_series.str.split('-', expand=True)
    return parts[0].astype(int) * 12 + parts[1].astype(int)


def build_comp_features(ref_df, target_df, lookbacks=(6, 12), include_price_stats=True):
    """
    For each row in target_df, look back N months in ref_df for the same
    (town, flat_model) group and compute comparable-sales statistics.

    ref_df             — training DataFrame; columns required:
                         town, flat_model, Tranc_YearMonth, resale_price, mid_storey
    target_df          — DataFrame to enrich (train or test).
                         Tranc_YearMonth required; resale_price not needed for test.
    lookbacks          — tuple of month window sizes, default (6, 12).
    include_price_stats — if True, also adds comp_6m_median, comp_12m_median,
                         comp_floor_adj_median. Set False to keep only comp_6m_count
                         (avoids collinearity with existing OOF price encodings).

    Always returns: comp_6m_count (# comparable sales in past 6 months)
    Also returns when include_price_stats=True:
        comp_6m_median, comp_12m_median, comp_floor_adj_median

    Leakage note: hi = searchsorted(months, m, side='left') excludes the current
    calendar month, so no same-period prices enter the comp window.
    """
    lb6, lb12 = lookbacks[0], lookbacks[1]

    tgt = target_df.copy()
    tgt['_m'] = _to_month_int(tgt['Tranc_YearMonth'])

    ref = ref_df[['town', 'flat_model', 'Tranc_YearMonth', 'resale_price', 'mid_storey']].copy()
    ref['_m'] = _to_month_int(ref['Tranc_YearMonth'])
    global_med = float(ref['resale_price'].median())

    # Pre-build sorted reference arrays per (town, flat_model) for O(log N) lookups
    ref_groups = {}
    for (town, fm), grp in ref.groupby(['town', 'flat_model']):
        s = grp['_m'].values.argsort()
        ref_groups[(town, fm)] = (
            grp['_m'].values[s],
            grp['resale_price'].values[s],
            grp['mid_storey'].values[s],
        )

    n = len(tgt)
    cnt6    = np.zeros(n, dtype=int)
    med6    = np.full(n, np.nan)
    med12   = np.full(n, np.nan)
    avg_st6 = np.full(n, np.nan)

    for (town, fm), grp_idx in tgt.groupby(['town', 'flat_model']).groups.items():
        if (town, fm) not in ref_groups:
            continue
        ref_months, ref_prices, ref_storeys = ref_groups[(town, fm)]
        tgt_months = tgt.loc[grp_idx, '_m'].values

        # Vectorised binary search — one call per group, not per row
        lo6  = np.searchsorted(ref_months, tgt_months - lb6,  side='left')
        lo12 = np.searchsorted(ref_months, tgt_months - lb12, side='left')
        hi   = np.searchsorted(ref_months, tgt_months,        side='left')

        for i, orig_idx in enumerate(grp_idx):
            p  = tgt.index.get_loc(orig_idx)
            w6 = ref_prices[lo6[i]:hi[i]]
            cnt6[p] = len(w6)
            if include_price_stats:
                med6[p]    = np.median(w6) if len(w6) else np.nan
                avg_st6[p] = np.mean(ref_storeys[lo6[i]:hi[i]]) if len(w6) else np.nan
                w12        = ref_prices[lo12[i]:hi[i]]
                med12[p]   = np.median(w12) if len(w12) else np.nan

    tgt['comp_6m_count'] = cnt6.astype(int)

    if include_price_stats:
        tgt['comp_6m_median']  = med6
        tgt['comp_12m_median'] = med12
        tgt['_avg_st6']        = avg_st6
        tgt['comp_floor_adj_median'] = (
            tgt['comp_6m_median']
            + (tgt['mid_storey'] - tgt['_avg_st6']) * FLOOR_PREMIUM
        )
        for col in ['comp_6m_median', 'comp_12m_median', 'comp_floor_adj_median']:
            town_fill = tgt.groupby('town')[col].transform('median')
            tgt[col]  = tgt[col].fillna(town_fill).fillna(global_med)
        tgt.drop(columns=['_avg_st6'], inplace=True)

    tgt.drop(columns=['_m'], inplace=True)
    return tgt
