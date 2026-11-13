# Supplementary numbers for TWCC: datum offsets, footprint windows, single-crest errors, Table S2 by area class
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'code/v3'); sys.dont_write_bytecode = True
from twcc import twcc, summ, LAB, BINS, tbin
FT = 0.3048; DH = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15])
X = pd.read_parquet('data/v3_exposure.parquet'); X['ab'] = X.ab.astype(str)
cP = ['P0'] + [f'P{d}' for d in DH[1:]]; VP = X[cP].values.astype(float)
def P(h, V=VP): hh = np.clip(h, 0, 15); return np.array([np.interp(a, DH, v) for a, v in zip(hh, V)])
print('datum offsets (summed residents / crest footprint): first, old rebuild, TWCC')
for dl in [-1, -0.5, 0, 0.5, 1]:
    base = P(np.full(len(X), max(dl, 0)))
    pc, pf, pg, pt = [np.clip(P(h * FT + dl) - base, 0, None) for h in (X.dh_peak_ft, X.dh_first_ft, X.dpk_gb, X.e_law)]
    print(dl, round(pf.sum() / pc.sum(), 3), round(pg.sum() / pc.sum(), 3), round(pt.sum() / pc.sum(), 3))
H = pd.read_parquet('data/hand_pop_radii.parquet'); Y = X[['site_no', 'dh_peak_ft', 'dh_first_ft', 'dpk_gb', 'e_law']].merge(H, on='site_no')
for r in (1, 3, 5):
    V = Y[[f'{c}_r{r}' for c in cP]].values.astype(float)
    pc, pf, pg, pt = [P(h * FT, V) - V[:, 0] for h in (Y.dh_peak_ft, Y.dh_first_ft, Y.dpk_gb, Y.e_law)]
    print('window', r, 'km: person-events M', round(pc.sum() / 1e6, 2), 'first', round(pf.sum() / pc.sum(), 3), 'old', round(pg.sum() / pc.sum(), 3), 'TWCC', round(pt.sum() / pc.sum(), 3))
x = X[X.P_crest > 10]; rt = x.P_tw / x.P_crest; rf = x.P_first / x.P_crest
print('single crests: median first', round(rf.median(), 2), 'median TWCC', round(rt.median(), 2))
print(pd.DataFrame({'w25': [((rt[x.ab == b] - 1).abs() <= .25).mean() for b in LAB], 'over': [(rt[x.ab == b] > 1.25).mean() for b in LAB],
                    'under': [(rt[x.ab == b] < .75).mean() for b in LAB]}, index=LAB).round(2).T.to_string())
# Table S2 rows
R = pd.read_parquet('data/v3_twcc.parquet'); S = pd.read_parquet('data/v3_twcc_storm.parquet')
def row(d, e): return [f'{summ(d[d.ab == b], e).w25 * 100:.0f} / {summ(d[d.ab == b], e).mae_m:.2f}' for b in LAB] + [f'{summ(d, e).w25 * 100:.0f} / {summ(d, e).mae_m:.2f}']
T = {'First clear look': row(R, 'e_look'), 'Image branch': row(R, 'e_img'), 'Prior branch (no image)': row(R, 'e_pri'),
     'TWCC, cell weights': row(R, 'e_tw'), 'TWCC, law': row(R, 'e_law')}
M = pd.read_parquet('data/rev_recon.parquet').reset_index(drop=True); M['ab'] = pd.cut(M.A, BINS, labels=LAB).astype(str)
for sh, nm in ((1, 'TWCC, wait overstated by 1 day'), (-1, 'TWCC, wait understated by 1 day')):
    Rs, _ = twcc(M, shift=sh); T[nm] = row(Rs, 'e_law')
T['First clear look after the radar storm day'] = row(S, 'e_look'); T['Image branch from the radar storm day'] = row(S, 'e_img'); T['TWCC from the radar storm day'] = row(S, 'e_law')
c80 = (R.dpk >= R.lo80) & (R.dpk <= R.hi80); wd = R.hi80 / R.lo80
T['Coverage of 80% intervals (%)'] = [f'{c80[R.ab == b].mean() * 100:.0f}' for b in LAB] + [f'{c80.mean() * 100:.0f}']
T['Width of 80% intervals (factor)'] = [f'{wd[R.ab == b].median():.1f}' for b in LAB] + ['']
for k, v in T.items(): print('| ' + k + ' | ' + ' | '.join(v) + ' |')
# TWCC skill by wait in the smallest and largest classes
R['tb'] = tbin(R.lag.values)
print(R.groupby(['ab', 'tb']).apply(lambda g: (abs(g.e_law / g.dpk - 1) <= .25).mean()).unstack().reindex(LAB).round(2).to_string())
print(R.groupby(['ab', 'tb']).apply(lambda g: (abs(g.e_img / g.dpk - 1) <= .25).mean()).unstack().reindex(LAB).round(2).to_string())
