# TWCC fed the gauge stage read directly at the first clear overpass, with input and target on one datum.
# Input  d_ov  = h(t_overpass) - h_pre          (15-minute stage, h_pre = lowest stage in the 72 h before the crest)
# Target d_pk  = h(t_crest)    - h_pre          (same record, same datum)
# Daily analogue on the same datum: d_day = mean 15-minute stage over the local calendar day of that overpass - h_pre.
# The recession term f = ln(e_img) - ln(d1), the law weight and the prior are the stored held-out values of the main
# evaluation (trained on daily, rated depths at other gauges); no crest information enters the input.
import numpy as np, pandas as pd, datetime as dt
FT = 0.3048
LAB = ['<30', '30-100', '100-300', '300-1k', '1k-3k', '3k-10k', '>10k']
T = np.load('data/terra.npy', mmap_mode='r'); A = np.load('data/aqua.npy', mmap_mode='r'); D = np.load('data/days.npy', allow_pickle=True)
di = {str(d)[:10]: i for i, d in enumerate(D)}
v = pd.read_parquet('data/usgs/iv.parquet'); v = v[v.param == '00065'].dropna(subset=['value'])
e = pd.read_csv('data/usgs/events.csv', dtype={'site_no': str})
R = pd.read_parquet('data/v3_twcc.parquet'); keep = set(zip(R.site_no, R.peak_date))
rows = []
for (site, pdate), g in v.groupby(['site_no', 'peak_date']):
    if (site, pdate) not in keep: continue
    r = e[(e.site_no == site) & (e.peak_date == pdate)].iloc[0]; g = g.sort_values('time_utc')
    t = (g.time_utc - pd.Timestamp(pdate, tz='UTC')).dt.total_seconds().values / 3600; h = g.value.values
    if len(h) < 50: continue
    ic = np.argmax(h); tc = t[ic]; hc = h[ic]; pre = (t < tc) & (t >= tc - 72); hb = np.min(h[pre]) if pre.sum() > 4 else np.min(h[:ic + 1])
    if hc - hb < 0.3: continue                      # same crest definition as the 392 fifteen-minute crests (iv_ana.py)
    iy = int((r.lat - 24.5) / 0.1); ix = int((r.lon + 125) / 0.1); d0 = dt.date.fromisoformat(pdate); first = None
    for o in range(0, 11):
        for loc, cube in ((10.5, T), (13.5, A)):
            tu = o * 24 + loc - r.lon / 15; k = di.get((d0 + dt.timedelta(o)).isoformat())
            f = np.nan if k is None or cube[k, iy, ix] == 255 else cube[k, iy, ix] / 200
            if f >= 0.5 and t[0] <= tu <= t[-1]: first = (o, tu); break
        if first: break
    if first is None: rows.append(dict(site_no=site, peak_date=pdate)); continue
    o, tu = first; a0 = o * 24 - r.lon / 15; sel = (t >= a0) & (t < a0 + 24)
    tg = np.arange(a0, a0 + 24, 0.25); tg = tg[(tg >= t[0]) & (tg <= t[-1])]
    rows.append(dict(site_no=site, peak_date=pdate, o=o, after=tu >= tc, hb=hb, d_pk=hc - hb, d_ov=np.interp(tu, t, h) - hb,
                     d_day=np.interp(tg, t, h).mean() - hb if len(tg) > 48 else np.nan, day_cover=len(tg) / 96))
V = pd.DataFrame(rows); Z = R.merge(V, on=['site_no', 'peak_date']); print('15-minute crests in the evaluation set', len(Z), 'with a clear overpass', Z.o.notna().sum())
Z = Z[Z.o.notna()].copy(); print('overpass on the first clear day of the daily analysis', (Z.o == Z.lag).mean())
print('daily analogue available', Z.d_day.notna().sum())
FLOOR = 0.05
Z['f'] = np.log(Z.e_img) - np.log(Z.d1)
def run(Z, col, floor=FLOOR):
    look = np.maximum(Z[col], floor); img = look * np.exp(Z.f); tw = np.exp(Z.w_law * np.log(img) + (1 - Z.w_law) * np.log(Z.e_pri))
    return look, img, tw
for col in ['d_ov', 'd_day']:
    Z[f'look_{col}'], Z[f'img_{col}'], Z[f'tw_{col}'] = run(Z, col)
Z['ab'] = pd.Categorical(Z.ab, LAB, ordered=True); Z['g3'] = pd.cut(Z.A, [0, 100, 1000, 1e7], labels=['<100', '100-1k', '>1k'])
def summ(d, est, tgt='d_pk'):
    r = d[est] / d[tgt]
    return pd.Series({'n': len(d), 'w25': (np.abs(r - 1) <= 0.25).mean(), 'mae_m': np.median(np.abs(d[est] - d[tgt])) * FT, 'med_ratio': r.median()})
E = [('first look, daily mean stage', 'look_d_day'), ('first look, overpass stage', 'look_d_ov'), ('image, overpass stage', 'img_d_ov'), ('prior', 'e_pri'),
     ('TWCC, daily mean stage', 'tw_d_day'), ('TWCC, overpass stage', 'tw_d_ov')]
def table(d, nm): o = pd.DataFrame({k: summ(d, c) for k, c in E}).T; o.insert(0, 'set', nm); return o
out = [table(Z, 'all')]; print('ALL', len(Z), 'floored overpass', int((Z.d_ov < FLOOR).sum())); print(out[0].round(3).to_string())
print('rated benchmark on the same crests (daily input, rated target):', summ(Z, 'e_law', 'dpk').round(3).to_dict(), 'first look', summ(Z, 'e_look', 'dpk').round(3).to_dict())
for col in ['ab', 'g3']:
    for b, d in Z.groupby(col, observed=True): o = table(d, f'{col}:{b}'); out.append(o)
P = pd.concat(out); print(P[P.set.str.startswith('g3')][['set', 'n', 'w25', 'mae_m', 'med_ratio']].round(2).to_string())
for nm, d in [('after crest', Z[Z.after]), ('before crest', Z[~Z.after])]:
    o = table(d, nm); out.append(o); print(nm, len(d)); print(o[['n', 'w25', 'mae_m', 'med_ratio']].round(3).to_string())
fl = Z.d_ov < FLOOR; print('floored crests', fl.sum(), 'TWCC overpass within 25% among them', summ(Z[fl], 'tw_d_ov').round(2).to_dict(), 'without them', summ(Z[~fl], 'tw_d_ov').round(3).to_dict())
for flo in [0.01, 0.1, 0.2]:
    _, _, tw = run(Z, 'd_ov', flo); print('floor', flo, 'ft: TWCC within 25%', round((np.abs(tw / Z.d_pk - 1) <= .25).mean(), 3))
rng = np.random.default_rng(0); sites = Z.site_no.unique(); idx = {s: np.where(Z.site_no.values == s)[0] for s in sites}
hd = (np.abs(Z.tw_d_day / Z.d_pk - 1) <= .25).values; ho = (np.abs(Z.tw_d_ov / Z.d_pk - 1) <= .25).values; hl = (np.abs(Z.look_d_ov / Z.d_pk - 1) <= .25).values
bs = []
for _ in range(2000):
    k = np.concatenate([idx[s] for s in rng.choice(sites, len(sites))]); bs.append([hd[k].mean(), ho[k].mean(), hl[k].mean(), ho[k].mean() - hl[k].mean()])
bs = np.array(bs); print('bootstrap 95%: daily', np.percentile(bs[:, 0], [2.5, 97.5]).round(3), 'overpass', np.percentile(bs[:, 1], [2.5, 97.5]).round(3),
                      'overpass-daily', np.percentile(bs[:, 1] - bs[:, 0], [2.5, 97.5]).round(3), 'gain over first look', np.percentile(bs[:, 3], [2.5, 97.5]).round(3))
# first-look depth fractions read three ways (Supplementary Section S2): rated daily flow, daily mean stage and overpass stage
Z['f_rated'] = Z.d1 / Z.dpk; Z['f_day'] = np.clip(Z.d_day, 0, None) / Z.d_pk; Z['f_ov'] = np.clip(Z.d_ov, 0, None) / Z.d_pk
def fr(d): return pd.Series({'n': len(d), 'rated': d.f_rated.median(), 'daily_stage': d.f_day.median(), 'overpass': d.f_ov.median(),
                             'absdiff_daily_overpass': (d.f_day - d.f_ov).abs().median(), 'absdiff_rated_overpass': (d.f_rated - d.f_ov).abs().median(),
                             'diff_rated_overpass': (d.f_rated - d.f_ov).median(), 'rated_above_overpass': (d.f_rated > d.f_ov).mean()})
F = pd.concat([Z.groupby('ab', observed=True).apply(fr), fr(Z).to_frame('all').T]); print(F.round(4).to_string()); F.to_csv('fig/rev/overpass_depth_fractions.csv')
# the same stages measured from the rated base (annual-peak gauge height minus the rated crest depth), for crests with a recorded peak gauge height
P = pd.read_parquet('data/usgs/peaks.parquet'); P['peak_date'] = P.peak_dt.astype(str); P = P.dropna(subset=['gage_ht_ft']).drop_duplicates(['site_no', 'peak_date'])
G = Z.merge(P[['site_no', 'peak_date', 'gage_ht_ft']], on=['site_no', 'peak_date'], how='left').gage_ht_ft.values
br = G - Z.dpk.values; hc15 = (Z.hb + Z.d_pk).values; m = np.isfinite(G)
print('crests with a peak gauge height', m.sum(), '| 15-minute crest stage equal to it', round((np.abs(hc15 - G)[m] <= 0.005).mean(), 3),
      '| 15-minute base above the rated base (ft, median)', round(np.median((Z.hb.values - br)[m]), 2), '| 15-minute over rated crest depth (median)', round((Z.d_pk / Z.dpk).median(), 3))
Z['f_ov_rb'] = (np.clip(Z.d_ov, 0, None) + Z.hb - br) / (hc15 - br); Z['f_day_rb'] = (np.clip(Z.d_day, 0, None) + Z.hb - br) / (hc15 - br)
def rb(d): return pd.Series({'n': d.f_ov_rb.notna().sum(), 'diff_rated_overpass': (d.f_rated - d.f_ov).median(), 'diff_rated_overpass_rated_base': (d.f_rated - d.f_ov_rb).median(),
                             'diff_rated_daily_rated_base': (d.f_rated - d.f_day_rb).median(), 'rated_above_overpass_rated_base': (d.f_rated > d.f_ov_rb)[d.f_ov_rb.notna()].mean()})
R = pd.concat([Z.groupby('ab', observed=True).apply(rb), rb(Z).to_frame('all').T]); print(R.round(3).to_string()); R.to_csv('fig/rev/overpass_depth_fractions_rated_base.csv')
bc = ~Z.after.astype(bool); print('first look before the crest', round(bc.mean(), 3), 'below 30 km2', round(bc[(Z.ab == '<30').values].mean(), 3))
h_ = [hd.sum(), ho.sum(), hl.sum()]; print('crests within 25%: daily stage', h_[0], 'overpass', h_[1], 'first look', h_[2], 'of', len(Z))
print('bootstrap 95% (%): overpass-daily', np.percentile((bs[:, 1] - bs[:, 0]) * 100, [2.5, 97.5]).round(2), 'gain', np.percentile(bs[:, 3] * 100, [2.5, 97.5]).round(2))
pd.concat(out).to_csv('fig/rev/twcc_overpass_direct.csv'); Z.to_parquet('data/v3_twcc_overpass_direct.parquet')
