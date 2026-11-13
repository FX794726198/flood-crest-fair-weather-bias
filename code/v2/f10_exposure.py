# Fig. 10  Residents in the crest footprints that the first clear look misses (USA stage-based, five Caravan regions)
from cstyle import *
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
DS = ['namerica', 'camelsgb', 'lamah', 'camelsbr', 'camelscl']
G = pd.read_parquet('data/global/exposure_caravan.parquet'); G = G[G.dataset != 'camelsaus']; G['ab'] = G.ab.astype(str)
TW = '#1B7F5C'
EX = pd.read_parquet('data/v3_exposure.parquet'); EX['ab'] = EX.ab.astype(str).str.replace('–', '-')
R = pd.read_csv('fig/exposure_global.csv'); HD = pd.read_csv('fig/rev/hand_delta.csv'); HR = pd.read_csv('fig/rev/hand_radius.csv')
def share(d, a='P_first'): return d[a].sum() / d.P_crest.sum() if d.P_crest.sum() > 0 else np.nan

H = 123.7; fig = figure(H)
# ---------------- a: map of the share of crest-footprint residents seen at the first clear look
a = ax_mm(fig, 9, 4, 94, 74.2)
basemap(a, (-128, 24, -57, 63)); a.set_aspect(1.0, adjustable='box')
B = G.groupby('basin_id').agg(lat=('lat', 'first'), lon=('lon', 'first'), ds=('dataset', 'first'), pc=('P_crest', 'sum'), pf=('P_first', 'sum'), n=('P_crest', 'size')).reset_index()
B = B[(B.pc >= 50) & (B.n >= 3)].copy(); B['s'] = B.pf / B.pc; B['ppc'] = B.pc / B.n; B = B.sort_values('ppc', ascending=False)
def msz(v): return 0.35 + 0.42 * np.clip(np.log10(v), 0.5, 5) ** 2
cm = ListedColormap(plt.get_cmap('RdYlBu')(np.linspace(0.02, 0.98, 10))); nb = BoundaryNorm(np.linspace(0, 1, 11), 10)
sc = a.scatter(B.lon, B.lat, c=B.s, s=msz(B.ppc), cmap=cm, norm=nb, lw=0.1, edgecolors='#505050', alpha=0.95, zorder=3, rasterized=True)
geo_ticks(a, [-120, -90, -60, -30, 0], [-40, -20, 0, 20, 40, 60])
for k, (x, y, ha) in {'namerica': (-96, 58.5, 'center'), 'camelsgb': (-9, 60.5, 'right'), 'lamah': (21.5, 42.0, 'right'), 'camelsbr': (-31, -9, 'left'),
                      'camelscl': (-66, -50, 'left')}.items():
    a.text(x, y, REGN[k], color=REG[k], fontsize=6.4, fontweight='bold', ha=ha, va='center', path_effects=HALO, zorder=6)
cb = cbar(fig, 13.5, 41.8, 21, 1.5, sc, ticks=[0, 0.5, 1], fs=6.2); cb.ax.set_xticklabels(['0', '0.5', '1'])
fig.text(24 / W, 1 - 39.9 / H, 'Residents seen at first\nclear look / at crest', ha='center', va='bottom', fontsize=6.2, linespacing=0.95)
for i, v in enumerate([1e2, 1e3, 1e4]):
    a.scatter(-116.5 + i * 8.5, -22, s=msz(v), c='#9A9A9A', lw=0.1, edgecolors='#505050', zorder=6)
    a.text(-116.5 + i * 8.5, -26, f'10$^{int(np.log10(v))}$', ha='center', va='top', fontsize=6.0)
a.text(-108, -17.5, 'Residents per crest', ha='center', va='bottom', fontsize=6.2)
north(a, 0.075, 0.07, size=0.07); scalebar(a, -112, -50, 2000, lat_ref=-50, label='2,000 km')

# ---------------- b: United States, stage-based reference and its checks
b = ax_mm(fig, 124, 4, 43, 28.3)
st = np.array([share(EX[EX.ab == k]) for k in LABK]); gb = np.array([share(EX[EX.ab == k], 'P_tw') for k in LABK])
ch = R[R.set == 'US chain (MODIS)'].set_index('ab').reindex(LABK).seen.values
alt = np.array([HD[(HD.delta != 0) & (HD.ab == k)].seen.tolist() + HR[(HR.radius_km != 5) & (HR.ab == k)].seen.tolist() for k in LABK])
band(b, XC, alt.min(1), alt.max(1), FIRST, 0.16)
b.plot(XC, st, '-', color=FIRST, lw=1.3, label='First clear look', zorder=4); b.plot(XC, st, 'o', color=FIRST, ms=2.8, mec='white', mew=0.35, zorder=5)
b.plot(XC, ch, '--', color=INK, lw=1.0, label='Flow chain, MODIS', zorder=3)
b.plot(XC, gb, '-', color=TW, lw=1.3, label='TWCC', zorder=4); b.plot(XC, gb, 'o', color=TW, ms=2.8, mec='white', mew=0.35, zorder=5)
ref(b); area_axis(b); b.set_ylim(0, 1.3); b.set_yticks([0, 0.5, 1.0]); b.set_ylabel('Residents / at crest')
b.legend(handles=[Line2D([], [], color=TW, lw=1.3), Line2D([], [], color=FIRST, lw=1.3), Patch(fc=FIRST, alpha=0.16, lw=0), Line2D([], [], color=INK, lw=1.0, ls='--')],
         labels=['TWCC', 'First clear look', 'Datum ±1 m, 1–3 km', 'Flow chain, MODIS'], loc='lower right', fontsize=5.8, handlelength=1.4,
         bbox_to_anchor=(1.04, -0.02), labelspacing=0.15)

# ---------------- c: five regions
c = ax_mm(fig, 124, 45.3, 43, 28.3)
for ds in DS:
    d = R[(R.set == ds) & (R.ab != 'all')].set_index('ab').reindex(LABK)
    d = d[d.n >= 30] if 'n' in d else d
    xs = XC[[LABK.index(k) for k in d.index]]
    c.plot(xs, d.seen, '-', color=REG[ds], lw=1.0, zorder=3); c.plot(xs, d.seen, 'o', color=REG[ds], ms=2.4, mec='white', mew=0.3, zorder=4)
d = R[(R.set == 'All Caravan') & (R.ab != 'all')].set_index('ab').reindex(LABK); c.plot(XC, d.seen, '-', color=INK, lw=2.0, zorder=5)
c.text(XC[6] * 1.3, d.seen.values[6], 'All', fontsize=FSS, color=INK, va='center')
area_axis(c); c.set_ylim(0, 1.0); c.set_yticks([0, 0.5, 1.0]); c.set_ylabel('Residents seen / at crest')

# ---------------- d: where the missed residents live
T2 = 86; H2 = 27
d = ax_mm(fig, 12, T2, 42, H2)
pc = np.array([G[G.ab == k].P_crest.sum() for k in LABK]) / 1e6; pf = np.array([G[G.ab == k].P_first.sum() for k in LABK]) / 1e6
d.bar(range(7), pf, color=REBUILT, width=0.72, lw=0, label='Seen'); d.bar(range(7), pc - pf, bottom=pf, color=FIRST, alpha=0.85, width=0.72, lw=0, label='Missed')
for x, a_, b_ in zip(range(7), pc, pf): d.text(x, a_ + 0.8, f'{1 - b_ / a_:.0%}', ha='center', va='bottom', fontsize=5.9, color=FIRST)
class_ticks(d); d.set_ylim(0, 58); d.set_yticks([0, 20, 40]); d.set_ylabel('Person-events (millions)')
sm = G[G.ab.isin(LABK[:3])]; yb = 52
d.plot([-0.36, -0.36, 2.36, 2.36], [yb - 1.5, yb, yb, yb - 1.5], '-', color=INK, lw=0.6)
d.text(-0.36, yb + 1.0, f'{sm.P_crest.sum() / G.P_crest.sum():.0%} of residents, {(sm.P_crest - sm.P_first).sum() / (G.P_crest - G.P_first).sum():.0%} of missed',
       ha='left', va='bottom', fontsize=5.9, color=INK)
d.legend(loc='upper right', fontsize=6.0, handlelength=1.0, bbox_to_anchor=(1.03, 0.8))

# ---------------- e: totals by region
e = ax_mm(fig, 77.5, T2, 29, H2)
rows = [('United States', EX.P_crest.sum(), share(EX), FIRST, 'USA stage')] + [(REGN[k], G[G.dataset == k].P_crest.sum(), share(G[G.dataset == k]), REG[k], '') for k in DS] + \
       [('All five', G.P_crest.sum(), share(G), INK, '')]
y = np.arange(len(rows))[::-1]
for yy, (nm, p, s_, col, _) in zip(y, rows):
    e.barh(yy, s_, color=col, height=0.68, lw=0); e.barh(yy, 1 - s_, left=s_, color=col, alpha=0.22, height=0.68, lw=0)
    e.text(s_ - 0.03, yy, f'{s_:.0%}', va='center', ha='right', fontsize=5.8, color='white', fontweight='bold')
    e.text(1.03, yy, f'{p / 1e6:.0f}' if p >= 1e7 else f'{p / 1e6:.1f}', va='center', ha='left', fontsize=5.8, color=INK2)
e.text(1.03, len(rows) - 0.35, 'M', fontsize=5.8, color=INK2, va='bottom', ha='left')
e.set_yticks(y); e.set_yticklabels([r[0] for r in rows], fontsize=6.3); e.tick_params(axis='y', length=0)
e.set_xlim(0, 1); e.set_xticks([0, 0.5, 1]); e.set_xticklabels(['0', '50%', '100%']); e.set_ylim(-0.6, len(rows) - 0.4)
e.set_xlabel('Residents seen'); spines(e, ('bottom',))

# ---------------- f: single crests within +-25% of the crest-footprint residents
f = ax_mm(fig, 124, T2, 43, H2)
x = EX[EX.P_crest > 10]
for col, nm, k_, mk in [(FIRST, 'First clear look', 'P_first', 'o'), (TW, 'TWCC', 'P_tw', 'o')]:
    yv = [(np.abs(x[x.ab == k][k_] / x[x.ab == k].P_crest - 1) <= 0.25).mean() for k in LABK]
    f.plot(XC, yv, '-', color=col, lw=1.3, label=nm, zorder=3); f.plot(XC, yv, mk, color=col, ms=2.8, mec='white', mew=0.35, zorder=4)
r_ = [x[x.ab == k].P_tw / x[x.ab == k].P_crest for k in LABK]
f.plot(XC, [(v > 1.25).mean() for v in r_], ':', color=TW, lw=1.0, label='TWCC, over')
f.plot(XC, [(v < 0.75).mean() for v in r_], '--', color=TW, lw=0.8, label='TWCC, under')
area_axis(f); f.set_ylim(0, 1.2); f.set_yticks([0, 0.5, 1.0]); f.set_ylabel('Crests within ±25%')
f.legend(loc='upper center', fontsize=5.6, handlelength=1.2, bbox_to_anchor=(0.5, 1.05), labelspacing=0.12, ncol=2, columnspacing=0.5, handletextpad=0.3)

for l_, x_, t_ in [('a', 1, 2), ('b', 113, 2), ('c', 113, 43.3), ('d', 1, T2 - 3), ('e', 58, T2 - 3), ('f', 113, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig10_exposure')
print('global', round(share(G), 3), round(G.P_crest.sum() / 1e6, 1), 'gauges', G.basin_id.nunique(), 'US', round(share(EX), 3), round(share(EX, 'P_tw'), 3))
