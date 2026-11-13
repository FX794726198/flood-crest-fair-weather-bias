# Fig. 6  The deficit holds across seasons, regions, flood sizes, overpasses, thresholds, storm forcing and Sentinel-2 scenes
from cstyle import *
from matplotlib.colors import LogNorm
s = pd.read_parquet('data/main_sample.parquet'); R1 = pd.read_csv('fig/fig1_bins.csv')
s['ab'] = aclass(s.drain_area_km2).astype(str)
m = pd.to_datetime(s.peak_date).dt.month; s['sea'] = np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ['DJF', 'MAM', 'JJA'], 'SON')
s['reg'] = np.where(s.lon < -100, 'W', 'E')
MR = pd.read_parquet('data/mrms_feat.parquet'); s = s.merge(MR[['site_no', 'peak_date', 'P_max24']], on=['site_no', 'peak_date'], how='left')
def r0(d, c='b'): return d[f'{c}0'].mean() / d[f'clim_{c}'].mean() if len(d) >= 30 else np.nan
def line(a, sel, col, lbl, mk='o', ls='-', key='b', lw=1.25):
    y = np.array([r0(s[sel & (s.ab == k)], key) for k in LABK])
    a.plot(XC, y, ls, color=col, lw=lw, label=lbl, zorder=3); a.plot(XC, y, mk, color=col, ms=2.8, mec='white', mew=0.35, zorder=4); return y
def base(a, ylim=(0.2, 1.25), xl=True):
    a.fill_between(XC, R1.r0l, R1.r0h, color=GREY_L, alpha=0.55, lw=0, zorder=0); ref(a); a.set_ylim(*ylim); area_axis(a, label=xl)
H = 98; fig = figure(H); T1, H1 = 3.5, 35; T2, H2 = 54, 35
T = np.ones(len(s), bool)

# ---------------- row 1: shared y axis
W1 = (167 - 12 - 3 * 2.5) / 4; L1 = [12 + i * (W1 + 2.5) for i in range(4)]
ax = [ax_mm(fig, l, T1, W1, H1) for l in L1]
for sea in ['DJF', 'MAM', 'JJA', 'SON']: line(ax[0], s.sea == sea, SEAS[sea], SEASN[sea])
base(ax[0]); ax[0].legend(loc='lower right', fontsize=6.1, ncol=2, columnspacing=0.6, handlelength=1.1, bbox_to_anchor=(1.04, -0.03))
for rg, col, nm in [('E', EAST, 'East of 100°W'), ('W', WEST, 'West of 100°W')]: line(ax[1], s.reg == rg, col, nm)
base(ax[1]); ax[1].legend(loc='lower right', fontsize=6.1, handlelength=1.1, bbox_to_anchor=(1.04, -0.03))
for (lo, hi), col, nm in [((0, 0.8), '#9ECAE1', '2–5 yr'), ((0.8, 0.9), '#4292C6', '5–10 yr'), ((0.9, 1.01), '#08306B', '≥10 yr')]:
    line(ax[2], s.weibull_p.between(lo, hi, inclusive='left'), col, nm)
base(ax[2]); ax[2].legend(loc='lower right', fontsize=6.1, handlelength=1.1, bbox_to_anchor=(1.04, -0.03), title='Flood size', title_fontsize=6.1)
line(ax[3], T, '#E69F00', 'Terra, 10:30', 'o', '-', 't'); line(ax[3], T, '#0072B2', 'Aqua, 13:30', 's', '-', 'a'); line(ax[3], T, INK, 'Better of two', 'D', '--', 'b', 1.0)
base(ax[3]); ax[3].legend(loc='lower right', fontsize=6.1, handlelength=1.4, bbox_to_anchor=(1.04, -0.03))
ax[0].set_ylabel('Crest-day clear sky / normal'); ax[0].set_yticks([0.4, 0.6, 0.8, 1.0])
for a in ax[1:]: a.set_yticks([0.4, 0.6, 0.8, 1.0]); a.set_yticklabels([])

# ---------------- row 2
W2 = [(167 - 12 - 3 * 12) / 4] * 4; L2 = [12 + i * (W2[0] + 12) for i in range(4)]
e, f, g, h = [ax_mm(fig, l, T2, w, H2) for l, w in zip(L2, W2)]
O = range(0, 11); B = np.stack([s[f'b{o}'].values for o in O], 1); Q = np.stack([s[f'q{o}'].values for o in O], 1)
for th, col in [(0.3, '#FDAE61'), (0.5, '#D7191C'), (0.7, '#A50F15'), (0.9, '#4D0008')]:
    hit = np.nan_to_num(B, nan=-1) >= th; k = np.where(hit.any(1), hit.argmax(1), -1); q = np.where(k >= 0, Q[np.arange(len(s)), np.clip(k, 0, None)], np.nan)
    y = [np.nanmedian(q[(s.ab == kk).values]) for kk in LABK]
    e.plot(XC, y, '-', color=col, lw=1.2, label=f'≥{th}', zorder=3); e.plot(XC, y, 'o', color=col, ms=2.6, mec='white', mew=0.35, zorder=4)
area_axis(e); e.set_ylim(0, 1.05); e.set_yticks([0, 0.25, 0.5, 0.75, 1]); e.set_ylabel('Flow left at first look')
e.legend(loc='lower right', fontsize=6.1, handlelength=1.1, title='Clear fraction', title_fontsize=6.1, bbox_to_anchor=(1.04, -0.03))
line(f, s.P_max24 >= 25, '#B2182B', '≥25 mm'); line(f, s.P_max24.between(5, 25, inclusive='left'), '#F4A582', '5–25 mm'); line(f, s.P_max24 < 5, '#2166AC', '<5 mm')
base(f, (0.2, 2.45)); f.set_yticks([0.5, 1.0, 1.5, 2.0]); f.set_ylabel('Crest-day clear sky / normal')
f.legend(loc='upper left', fontsize=6.1, handlelength=1.1, title='Rain (MRMS)', title_fontsize=6.1, bbox_to_anchor=(-0.02, 1.04))

# g: Sentinel-2 scene classification against the GOES-16 clear fraction (same day, Terra time)
S2 = pd.read_parquet('data/s2/s2_scl.parquet'); S2 = S2[(S2.day >= 0) & (S2.cov1 >= 0.9) & S2.goes_t.notna()].copy()
S2['abk'] = S2.ab.astype(str)
for gsel, col, nm in [(True, '#2166AC', 'GOES-16 clear'), (False, GREY, 'GOES-16 cloudy')]:
    sel = (S2.goes_t >= 0.5) if gsel else (S2.goes_t < 0.5)
    y = [(S2[sel & (S2.abk == k)].clr1 >= 0.5).mean() for k in LABK]
    g.plot(XC, y, '-', color=col, lw=1.25, label=nm, zorder=3); g.plot(XC, y, 'o', color=col, ms=2.8, mec='white', mew=0.35, zorder=4)
agree = ((S2.goes_t >= 0.5) == (S2.clr1 >= 0.5)).mean()
area_axis(g); g.set_ylim(0, 1.0); g.set_yticks([0, 0.25, 0.5, 0.75, 1]); g.set_ylabel('Sentinel-2 view cloud-free')
g.legend(loc='center right', fontsize=6.1, handlelength=1.1, bbox_to_anchor=(1.05, 0.52))
print('agreement', round(agree * 100, 1))

# h: usable Sentinel-2 scenes within one day of the crest and later
for early, col, nm in [(True, OBS, 'Crest day or next'), (False, GREY, 'Later')]:
    y = [((S2.abk == k) & (S2.day <= 1 if early else S2.day >= 2) & (S2.clr1 >= 0.5)).sum() / ((S2.abk == k) & (S2.day <= 1 if early else S2.day >= 2)).sum() for k in LABK]
    h.plot(XC, y, '-', color=col, lw=1.25, label=nm, zorder=3); h.plot(XC, y, 'o', color=col, ms=2.8, mec='white', mew=0.35, zorder=4)
area_axis(h); h.set_ylim(0, 0.62); h.set_yticks([0, 0.2, 0.4, 0.6]); h.set_ylabel('Cloud-free Sentinel-2 scenes')
h.legend(loc='lower right', fontsize=6.1, handlelength=1.1, bbox_to_anchor=(1.05, -0.03))

for l_, x_, t_ in [('a', 1, 1.5)] + [(n_, L1[i] - 3.5, 1.5) for i, n_ in [(1, 'b'), (2, 'c'), (3, 'd')]] + [(n_, L2[i] - 11, T2 - 3) for i, n_ in enumerate('efgh')]: letter(fig, l_, x_, t_)
save(fig, 'Fig06_robustness')
