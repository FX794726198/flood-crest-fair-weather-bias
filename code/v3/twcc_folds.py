# Export the gauge-grouped outer and inner fold assignments used by twcc.py (GroupKFold is deterministic for a fixed row order)
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
def folds(M, name):
    M = M.reset_index(drop=True); out = M[['site_no', 'peak_date']].copy(); out['outer'] = -1
    for k in range(5): out[f'inner_in_outer{k}'] = -1
    for k, (tr, te) in enumerate(GroupKFold(5).split(M, M.index, M.site_no)):
        out.loc[te, 'outer'] = k; T = M.iloc[tr].reset_index(drop=True)
        for j, (_, b) in enumerate(GroupKFold(5).split(T, T.index, T.site_no)): out.loc[np.asarray(tr)[b], f'inner_in_outer{k}'] = j
    assert (out.outer >= 0).all(); out.to_csv(name, index=False); print(name, len(out), out.outer.value_counts().sort_index().to_dict())
folds(pd.read_parquet('data/rev_recon.parquet'), 'data/twcc_folds_daily.csv')
folds(pd.read_parquet('data/rev_recon_storm.parquet'), 'data/twcc_folds_storm.csv')
