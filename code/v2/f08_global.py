# Fig. 8  The same pattern in six regions on four continents (Caravan x MODIS 1 deg x ERA5-Land)
from cstyle import *
from matplotlib.colors import BoundaryNorm, ListedColormap
C = pd.read_parquet('data/global/caravan_ana.parquet'); U = pd.read_parquet('data/global/usgs_matched.parquet'); U = U[U.weibull_p >= 0.5]
G = pd.read_csv('fig/fig4_cells.csv'); CU = pd.read_csv('fig/fig4_curves.csv', index_col=0); A1 = pd.read_csv('fig/fig4_area_by_dataset.csv')
UGM = pd.read_parquet('data/global/usgs_goes_modis.parquet'); UGM['ab'] = aclass(UGM.drain_area_km2)
X1 = pd.read_csv('fig/rev/mech_crossfit_global.csv'); LG = pd.read_csv('fig/rev/mech_area_by_lag_global.csv')
DS = ['namerica', 'camelsgb', 'lamah', 'camelsbr', 'camelscl', 'camelsaus']
def r0(d): return 0.5 * (d.T_0.mean() / d.clim_T.mean() + d.A_0.mean() / d.clim_A.mean())
def rg(d): return 0.5 * (d.t0.mean() / d.clim_t.mean() + d.a0.mean() / d.clim_a.mean())
JP = CU.index.values
def t75(cv):
    for k in range(np.where(JP == 0)[0][0], len(JP) - 1):
        if cv[k] < 0.75 <= cv[k + 1]: return JP[k] + (0.75 - cv[k]) / (cv[k + 1] - cv[k])
    return np.nan

H = 105; fig = figure(H); T2 = 59; H2 = 34
# ---------------- a: map of crest-day clear sky in 2-degree cells
a = ax_mm(fig, 9, 4, 112, 43.5)
basemap(a, (-170, 160, -56, 72)); a.set_aspect(1.0, adjustable='box')
X = pd.concat([C[['lat', 'lon', 'T_0', 'A_0', 'clim_T', 'clim_A']], U[['lat', 'lon', 'T_0', 'A_0', 'clim_T', 'clim_A']]])
X['ci'] = np.floor(X.lat / 2) * 2 + 1; X['cj'] = np.floor(X.lon / 2) * 2 + 1
M = X.groupby(['ci', 'cj']).apply(lambda d: pd.Series({'n': len(d), 'r': r0(d)})).reset_index(); M = M[M.n >= 20]
cm = ListedColormap(plt.get_cmap('RdYlBu')(np.linspace(0.02, 0.98, 10))); nb = BoundaryNorm(np.linspace(0.1, 1.1, 11), 10)
sc = a.scatter(M.cj, M.ci, c=M.r.clip(0.1, 1.099), s=4.2, marker='s', cmap=cm, norm=nb, lw=0, zorder=3, rasterized=True)
geo_ticks(a, [-150, -100, -50, 0, 50, 100, 150], [-40, -20, 0, 20, 40, 60])
for k, (x, y, ha) in {'namerica': (-128, 62, 'center'), 'camelsgb': (-12, 63, 'right'), 'lamah': (22, 52, 'left'), 'camelsbr': (-30, -12, 'left'),
                      'camelscl': (-66, -52, 'left'), 'camelsaus': (110, -15, 'right')}.items():
    a.text(x, y, REGN[k], color=REG[k], fontsize=6.4, fontweight='bold', ha=ha, va='center', path_effects=HALO, zorder=6)
north(a, 0.745, 0.22); scalebar(a, 52, -48, 2000, lat_ref=-48, label='2,000 km')
cb = cbar(fig, 14, 42.5, 20, 1.5, sc, ticks=[0.1, 0.6, 1.1], fs=6.2)
fig.text(24 / W, 1 - 40.8 / H, 'Crest-day clear sky / normal', ha='center', va='bottom', fontsize=6.2)

# ---------------- b: area dependence by region
b = ax_mm(fig, 135, 6, 32, 38)
for ds in DS:
    d = A1[(A1.dataset == ds) & (A1.n >= 40)].set_index('ab').reindex(LABK)
    b.plot(XC, d.r0, '-', color=REG[ds], lw=1.0, zorder=3); b.plot(XC, d.r0, 'o', color=REG[ds], ms=2.4, mec='white', mew=0.3, zorder=4)
allr = [r0(C[C.rain & (C.ab == k)]) for k in LABK]; b.plot(XC, allr, '-', color=INK, lw=2.0, zorder=5)
b.text(XC[6] * 1.25, allr[6], 'All', fontsize=FSS, color=INK, va='center', ha='left')
area_axis(b); b.set_ylim(0, 1.0); b.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); b.set_ylabel('Crest-day clear sky / normal')

# ---------------- c: storm-aligned clear sky by region
c = ax_mm(fig, 12, T2, 31, H2)
c.axvspan(-0.5, 0.5, color=GREY_LL, lw=0, zorder=0)
for ds in DS: c.plot(JP, CU[ds], '-', color=REG[ds], lw=1.0)
c.plot(JP, CU['all'], '-', color=INK, lw=2.0); ref(c); c.axhline(0.75, color=INK2, lw=0.6, ls=':')
c.set_xlim(-2, 7); c.set_xticks([-2, 0, 2, 4, 6]); c.set_ylim(0, 1.3); c.set_yticks([0, 0.5, 1.0])
c.set_xlabel('Days after storm day'); c.set_ylabel('Clear sky / normal'); c.text(0, 1.28, 'Storm', ha='center', va='top', fontsize=FSS, color=INK2)
tt = {k: t75(CU[k].values) for k in DS}
print('t75', {k: round(v, 2) for k, v in tt.items()})

# ---------------- d: rain and no-rain crests; MODIS against GOES-16 on the same USGS crests
d = ax_mm(fig, 55, T2, 31, H2)
A0 = C[C.Pmax < 5]; Ar = C[C.rain]
for d0, col, nm, mk in [(Ar, OBS, 'Storm-driven', 'o'), (A0, '#2166AC', 'No rain', 's')]:
    y = [r0(d0[d0.ab == k]) for k in LABK]; d.plot(XC, y, '-', color=col, lw=1.25, label=nm, zorder=3); d.plot(XC, y, mk, color=col, ms=2.7, mec='white', mew=0.35, zorder=4)
d.plot(XC, [rg(UGM[UGM.ab == k]) for k in LABK], '--', color=INK2, lw=1.0, label='USA, GOES-16')
d.plot(XC, [r0(UGM[UGM.ab == k]) for k in LABK], ':', color=INK, lw=1.3, label='USA, MODIS')
ref(d); area_axis(d); d.set_ylim(0, 1.32); d.set_yticks([0, 0.5, 1.0]); d.set_ylabel('Crest-day clear sky / normal')
d.legend(loc='lower right', fontsize=5.9, handlelength=1.5, bbox_to_anchor=(1.05, -0.03), labelspacing=0.2)

# ---------------- e: cross-fitted prediction
e = ax_mm(fig, 98, T2, 30, H2)
for ds in DS:
    q = G[G.dataset == ds]; e.scatter(q.pred, q.obs, s=5 + q.n / 30, color=REG[ds], edgecolor=INK, lw=0.25, alpha=0.9, zorder=3)
r = np.corrcoef(G.obs, G.pred)[0, 1]; rm = np.sqrt(((G.obs - G.pred) ** 2).mean()); xb = X1.set_index('scheme')
e.plot([0, 1.3], [0, 1.3], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1); e.set_xlim(0, 1.05); e.set_ylim(0, 1.05); e.set_xticks([0, 0.5, 1]); e.set_yticks([0, 0.5, 1])
stat(e, f'r = {r:.2f}, RMSE {rm:.2f}', 0.03, 0.99, fs=6.0)
print('8e', round(r, 3), round(rm, 3), len(G), xb.loc[['2.5° blocks', 'storms', 'years'], 'r'].round(3).tolist(), round(xb.r_null.mean(), 3))
e.legend([e.scatter([], [], s=5 + n_ / 30, color='white', edgecolor=INK, lw=0.5) for n_ in (100, 500, 2000)], ['100', '500', '2,000'], title='Crests', title_fontsize=6.1,
         loc='lower right', fontsize=6.1, handletextpad=0.3, labelspacing=0.45, borderaxespad=0.1, bbox_to_anchor=(1.03, -0.02))
e.set_xlabel('Predicted'); e.set_ylabel('Observed clear sky / normal')

# ---------------- f: lag within each area class
f = ax_mm(fig, 139, T2, 28, H2)
for i, k in enumerate(LABK):
    q = LG[LG.ab == k].sort_values('Lc'); q = q[q.n >= 40]
    f.plot(q.Lc, q.r0, '-', color=ACOL[i], lw=1.1, zorder=3); f.plot(q.Lc, q.r0, 'o', color=ACOL[i], ms=2.6, mec='white', mew=0.3, zorder=4)
ref(f); f.set_xticks([0, 1, 2, 3]); f.set_xticklabels(['0', '1', '2', '≥3']); f.set_xlim(-0.25, 3.25); f.set_ylim(0, 1.1); f.set_yticks([0, 0.5, 1.0])
f.set_xlabel('Crest lag (days)'); f.set_ylabel('Crest-day clear sky / normal')

for l_, x_, t_ in [('a', 1, 2), ('b', 124, 2), ('c', 1, T2 - 3), ('d', 44, T2 - 3), ('e', 87, T2 - 3), ('f', 128, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig08_global', ignore=('r = 0.91', "'f'"))
print('t75', {k: round(v, 2) for k, v in tt.items()}, 'r', round(r, 3))
