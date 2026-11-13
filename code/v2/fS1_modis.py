# Fig. S1  MODIS 1-degree cloud fraction reproduces the GOES-16 0.1-degree crest-day deficit and first clear look on the same USGS crests
from cstyle import *
from matplotlib.colors import LogNorm
U = pd.read_parquet('data/global/usgs_goes_modis.parquet'); U['ab'] = U.ab.astype(str); SUB = pd.read_csv('fig/rev/S1_subset.csv').set_index('ab').reindex(LABK)
key = lambda o: '0' if o == 0 else (f'm{-o}' if o < 0 else f'p{o}')
H = 52; fig = figure(H); T, HH = 4, 36
# ---------------- a
a = ax_mm(fig, 12, T, 40, HH)
for col, nm, ls, mk, v in [(OBS, 'GOES-16, 0.1°', '-', 'o', SUB.goes01_sub1), ('#EF8A62', 'GOES-16, 1°', '--', 's', SUB.goes1), (PRED, 'MODIS, 1°', '-', 'D', SUB.modis)]:
    a.plot(XC, v.values, ls, color=col, lw=1.2, label=nm, zorder=3); a.plot(XC, v.values, mk, color=col, ms=2.8, mec='white', mew=0.35, zorder=4)
ref(a); area_axis(a); a.set_ylim(0.2, 1.1); a.set_yticks([0.4, 0.6, 0.8, 1.0]); a.set_ylabel('Crest-day clear sky / normal')
a.legend(loc='lower right', fontsize=6.1, bbox_to_anchor=(1.03, -0.02), handlelength=1.6)
# ---------------- b
b = ax_mm(fig, 68, T, 40, HH)
O = range(0, 11); Q = np.stack([U[f'q{o}'].values for o in O], 1)
def first(B, th):
    hit = np.nan_to_num(B, nan=-1) >= th; k = np.where(hit.any(1), hit.argmax(1), -1); return np.where(k >= 0, Q[np.arange(len(U)), np.clip(k, 0, None)], np.nan)
BG = np.stack([U[f'b{o}'].values for o in O], 1); BM = np.stack([np.fmax(U[f'T_{key(o)}'].values, U[f'A_{key(o)}'].values) for o in O], 1)
qg = first(BG, 0.5); y = [np.nanmedian(qg[(U.ab == k).values]) for k in LABK]
b.plot(XC, y, '-', color=OBS, lw=1.4, label='GOES-16 ≥ 0.5', zorder=4); b.plot(XC, y, 'o', color=OBS, ms=2.8, mec='white', mew=0.35, zorder=5)
for th, col, ls in [(0.3, '#9ECAE1', ':'), (0.4, PRED, '-'), (0.5, '#08306B', ':')]:
    qm = first(BM, th); y = [np.nanmedian(qm[(U.ab == k).values]) for k in LABK]
    b.plot(XC, y, ls, color=col, lw=1.1, label=f'MODIS ≥ {th}', zorder=3); b.plot(XC, y, 'D', color=col, ms=2.4, mec='white', mew=0.3, zorder=4)
area_axis(b); b.set_ylim(0, 1.05); b.set_yticks([0, 0.25, 0.5, 0.75, 1]); b.set_ylabel('Flow left at first clear look')
b.legend(loc='upper left', fontsize=6.0, bbox_to_anchor=(0.03, 1.04), handlelength=1.6, labelspacing=0.18)
# ---------------- c
c = ax_mm(fig, 124, T, 34, HH)
g = U.b0.values; m = np.fmax(U.T_0.values, U.A_0.values); ok = np.isfinite(g) & np.isfinite(m)
h = c.hist2d(g[ok], m[ok], bins=[np.linspace(0, 1, 11), np.linspace(0, 1, 11)], cmap='Blues', cmin=1, norm=LogNorm(), rasterized=True)
r = np.corrcoef(g[ok], m[ok])[0, 1]; c.plot([0, 1], [0, 1], '--', color=INK, lw=0.6)
stat(c, f'r = {r:.2f}', 0.04, 0.96, fs=6.2, bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.8)); print('S1 n', len(U), int(ok.sum()))
c.set_xlim(0, 1); c.set_ylim(0, 1); c.set_xticks([0, 0.5, 1]); c.set_yticks([0, 0.5, 1]); c.set_aspect('equal', adjustable='box')
c.set_xlabel('GOES-16 clear, crest day'); c.set_ylabel('MODIS clear, crest day')
cb = cbar(fig, 160, T + 2, 1.8, HH - 4, h[3], orient='vertical', fs=6.0); cb.set_label('Crests', fontsize=6.2, labelpad=1)
for l_, x_ in [('a', 1), ('b', 57), ('c', 113)]: letter(fig, l_, x_, T - 2)
save(fig, 'FigS1_modis')
print('r', round(r, 3), 'n', ok.sum())
