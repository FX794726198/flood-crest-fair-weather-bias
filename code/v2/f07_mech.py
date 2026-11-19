# Fig. 7  Crest lag and cloud clearing set the deficit (United States, MRMS storms)
from cstyle import *
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
rng = np.random.default_rng(1)
M = pd.read_parquet('data/mech.parquet'); M['L'] = M.lag + 1
M['ab'] = aclass(M.drain_area_km2); M['reg'] = np.where(M.lon < -100, 'W', 'E')
JP = np.arange(-2, 8)
B = np.stack([M[f'b{o}'].values for o in range(-3, 11)], 1); o = M.kr.values[:, None] + JP[None, :] - 1; ok = (o >= -3) & (o <= 10)
E = np.where(ok, np.take_along_axis(B, np.clip(o + 3, 0, 13), 1), np.nan); C = M.clim_b.values
def curve(idx):
    e = E[idx]; c = np.broadcast_to(C[idx, None], e.shape); m = np.isfinite(e); return np.where(m, e, 0).sum(0) / np.where(m, c, 0).sum(0)
usite, inv = np.unique(M.site_no.values, return_inverse=True); members = [np.where(inv == k)[0] for k in range(len(usite))]
def boot_idx(idx):
    s_ = np.unique(inv[idx]); ss = rng.choice(s_, len(s_)); return np.concatenate([np.intersect1d(members[k], idx, assume_unique=True) for k in ss])
curves = {k: curve(np.where(M.ab == k)[0]) for k in LABK}
RC = {}
for rg in 'EW':
    idx = np.where(M.reg == rg)[0]; bs = np.array([curve(boot_idx(idx)) for _ in range(200)]); RC[rg] = (curve(idx), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0))
TC = {'E': (1.14, 1.09, 1.17), 'W': (2.02, 1.77, 2.25)}          # cloud-clearing time and 95% interval (fig3.py, 200 bootstraps of gauges)
R = pd.read_csv('fig/fig3_lag_by_area_region.csv'); G = pd.read_csv('fig/fig3_cells.csv'); X = pd.read_csv('fig/rev/mech_crossfit_US.csv')
LG = pd.read_csv('fig/rev/mech_area_by_lag_US.csv')

H = 120; fig = figure(H); T2 = 73; H2 = 34
# ---------------- a: map of mean crest lag
a = ax_mm(fig, 9, 4, 101, 55.5)
basemap(a, (-125, -66.5, 24.5, 49.8), states=True); a.set_aspect(1 / np.cos(np.deg2rad(38)), adjustable='box')
p = M.groupby('site_no').agg(lat=('lat', 'first'), lon=('lon', 'first'), L=('L', 'mean'), A=('drain_area_km2', 'first')).sort_values('L')
cm = ListedColormap(plt.get_cmap('viridis_r')(np.linspace(0.05, 0.95, 6))); nb = BoundaryNorm([0, 0.5, 1, 1.5, 2, 2.5, 3], 6)
sc = a.scatter(p.lon, p.lat, c=p.L.clip(0, 2.999), s=4.6, cmap=cm, norm=nb, lw=0.15, edgecolors='#404040', rasterized=True, zorder=3)
geo_ticks(a, [-120, -110, -100, -90, -80, -70], [30, 35, 40, 45]); north(a, 0.935, 0.42); scalebar(a, -123.5, 26.3, 500, lat_ref=30)
a.axvline(-100, color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=2)
cb = cbar(fig, 91, 49.5, 16, 1.6, sc, ticks=[0, 1, 2, 3], fs=6.2); cb.ax.set_xticklabels(['0', '1', '2', '≥3'])
fig.text(99 / W, 1 - 47.6 / H, 'Mean crest lag after\nstorm day (days)', ha='center', va='bottom', fontsize=6.2, linespacing=0.95)

# ---------------- b: crest lag distribution by area class (class colours shown on the axis = legend for c-f)
b = ax_mm(fig, 138, 9, 29, 44)
T = pd.crosstab(M.ab, M.L.clip(0, 4), normalize='index').reindex(LABK)
lc = plt.get_cmap('viridis_r')(np.linspace(0.05, 0.95, 5)); left = np.zeros(7)
for k in range(5):
    b.barh(np.arange(7), T[k].values, left=left, color=lc[k], height=0.74, lw=0, label=['0', '1', '2', '3', '≥4'][k]); left += T[k].values
for i, v in enumerate(T[0].values):
    if v >= 0.12: b.text(v / 2, i, f'{v*100:.0f}', ha='center', va='center', fontsize=6.0, color=INK)
b.set_yticks(range(7)); b.set_yticklabels(LAB); b.invert_yaxis(); b.set_xlim(0, 1); b.set_xticks([0, 0.5, 1]); b.set_xlabel('Share of crests')
b.tick_params(axis='y', length=0, pad=13)
for i in range(7): b.add_patch(plt.Rectangle((-0.13, i - 0.3), 0.085, 0.6, color=ACOL[i], clip_on=False, transform=b.transData))
b.set_ylabel('Drainage area (km$^2$)', labelpad=1)
b.legend(title='Crest lag (days)', ncol=5, fontsize=6.1, title_fontsize=6.3, loc='lower center', bbox_to_anchor=(0.36, 1.0), handlelength=0.9, columnspacing=0.5, handletextpad=0.3)
spines(b, ('bottom',))

# ---------------- c: storm-aligned clear sky
c = ax_mm(fig, 12, T2, 31, H2)
c.axvspan(-0.5, 0.5, color=GREY_LL, lw=0, zorder=0)
for i, k in enumerate(LABK): c.plot(JP, curves[k], '-', color=ACOL[i], lw=0.7, alpha=0.9, zorder=2)
for rg, ls, nm, col in [('E', '-', 'East', EAST), ('W', '--', 'West', WEST)]:
    c.fill_between(JP, RC[rg][1], RC[rg][2], color=col, alpha=0.18, lw=0); c.plot(JP, RC[rg][0], color=col, ls=ls, lw=1.6, label=nm, zorder=4)
ref(c); c.axhline(0.75, color=INK2, lw=0.6, ls=':'); c.set_xlim(-2, 7); c.set_xticks([-2, 0, 2, 4, 6]); c.set_ylim(0, 1.45); c.set_yticks([0, 0.5, 1.0])
c.set_xlabel('Days after storm day'); c.set_ylabel('Clear sky / normal')
c.legend(loc='lower right', fontsize=6.1, handlelength=1.6, bbox_to_anchor=(1.04, -0.03)); c.text(0, 1.42, 'Storm', ha='center', va='top', fontsize=FSS, color=INK2)

# ---------------- d: two timescales against area
d = ax_mm(fig, 55, T2, 31, H2)
for rg, mk, ls, col in [('E', 'o', '-', EAST), ('W', '^', '--', WEST)]:
    t, tl, th = TC[rg]; d.axhspan(tl, th, color=col, alpha=0.15, lw=0); d.axhline(t, color=col, ls=ls, lw=1.0)
    q = R[R.reg == rg]; d.fill_between(q.A, q.Ll, q.Lh, color=GREY, alpha=0.2, lw=0); d.plot(q.A, q.Lmean, ls, color=INK, lw=1.0)
    for x, y, k in zip(q.A, q.Lmean, q.b): d.plot(x, y, mk, color=ACOL[LABK.index(k)], mec=INK, mew=0.35, ms=3.6, zorder=5)
d.text(8, 1.14 + 0.07, 'Clearing (E)', fontsize=FSS, color=EAST, va='bottom'); d.text(8, 2.02 + 0.07, 'Clearing (W)', fontsize=FSS, color=WEST, va='bottom')
d.legend([Line2D([], [], color=INK, ls='-', lw=1.0, marker='o', ms=3.4, mfc='white', mec=INK, mew=0.5), Line2D([], [], color=INK, ls='--', lw=1.0, marker='^', ms=3.4, mfc='white', mec=INK, mew=0.5)], ['East', 'West'],
         title='Crest lag', title_fontsize=6.1, loc='lower right', fontsize=6.1, handlelength=2.2, borderaxespad=0.1, bbox_to_anchor=(1.03, -0.02))
area_axis(d, lim=(7, 4e4)); d.set_ylim(0, 3.0); d.set_yticks([0, 1, 2, 3]); d.set_ylabel('Days after storm day')

# ---------------- e: cross-fitted prediction
e = ax_mm(fig, 98, T2, 30, H2)
for i, k in enumerate(LABK):
    for rg, mk in [('E', 'o'), ('W', '^')]:
        q = G[(G.b == k) & (G.reg == rg)]; e.scatter(q.pred, q.obs, s=6 + q.n / 18, marker=mk, color=ACOL[i], edgecolor=INK, lw=0.3, alpha=0.95, zorder=3)
e.plot([0, 1.8], [0, 1.8], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1); e.set_xlim(0, 1.65); e.set_ylim(0, 1.65); e.set_xticks([0, 0.5, 1, 1.5]); e.set_yticks([0, 0.5, 1, 1.5])
r = np.corrcoef(G.obs, G.pred)[0, 1]; rm = np.sqrt(((G.obs - G.pred) ** 2).mean())
xb = X.set_index('scheme')
stat(e, f'r = {r:.2f}, RMSE {rm:.2f}', 0.03, 0.99, fs=6.0)
print('7e', round(r, 3), round(rm, 3), len(G), xb.loc[['2.5° blocks', 'storms', 'years'], 'r_lag'].round(3).tolist(), round(xb.loc['site halves', 'r_null'], 3))
e.legend([Line2D([], [], ls='', marker='o', ms=3.4, mfc='white', mec=INK, mew=0.6), Line2D([], [], ls='', marker='^', ms=3.4, mfc='white', mec=INK, mew=0.6)], ['East', 'West'],
         loc='lower right', fontsize=6.1, handletextpad=0.1, borderaxespad=0.1, bbox_to_anchor=(1.03, -0.02))
e.add_artist(e.get_legend())
e.legend([e.scatter([], [], s=6 + n_ / 18, marker='o', color='white', edgecolor=INK, lw=0.5) for n_ in (50, 200, 500)], ['50', '200', '500'], title='Crests', title_fontsize=6.1,
         loc='upper left', fontsize=6.1, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.1, bbox_to_anchor=(-0.01, 0.91))
e.set_xlabel('Predicted'); e.set_ylabel('Observed clear sky / normal')

# ---------------- f: lag within each area class
f = ax_mm(fig, 139, T2, 28, H2)
for i, k in enumerate(LABK):
    q = LG[LG.ab == k].sort_values('Lc'); q = q[q.n >= 15]
    f.plot(q.Lc, q.r0, '-', color=ACOL[i], lw=1.1, zorder=3); f.plot(q.Lc, q.r0, 'o', color=ACOL[i], ms=2.6, mec='white', mew=0.3, zorder=4)
ref(f); f.set_xticks([0, 1, 2, 3]); f.set_xticklabels(['0', '1', '2', '≥3']); f.set_xlim(-0.25, 3.25); f.set_ylim(0, 1.4); f.set_yticks([0, 0.5, 1.0])
f.set_xlabel('Crest lag (days)'); f.set_ylabel('Crest-day clear sky / normal')

for l_, x_, t_ in [('a', 1, 2), ('b', 114, 2), ('c', 1, T2 - 3), ('d', 44, T2 - 3), ('e', 87, T2 - 3), ('f', 128, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig07_mechanism', ignore=('r = 0.94', "'f'"))
print('r', round(r, 3), 'rmse', round(rm, 3))
