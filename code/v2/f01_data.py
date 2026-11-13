# Fig. 2  Data and study domain
from cstyle import *
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
s = pd.read_parquet('data/main_sample.parquet'); s['ab'] = aclass(s.drain_area_km2)
C = pd.read_parquet('data/global/caravan_ana.parquet')
DS = ['namerica', 'camelsgb', 'lamah', 'camelsbr', 'camelscl', 'camelsaus']
clim = np.load('data/clim_month.npy', mmap_mode='r'); g01 = np.nanmean(np.array(clim[2], dtype='f4'), 0)
mT = np.load('data/global/clim_T.npy').astype('f4'); mA = np.load('data/global/clim_A.npy').astype('f4'); m1 = np.nanmean(0.5 * (mT + mA), 0); m1[~np.load('data/landmask_1deg.npy')] = np.nan
cm_bg = ListedColormap(plt.get_cmap('Greys_r')(np.linspace(0.18, 0.92, 8))); nb_bg = BoundaryNorm(np.linspace(0.1, 0.9, 9), 8)

H = 127.5; fig = figure(H)
# ---------------- a: USGS gauges over the GOES-16 clear-sky climatology
a = ax_mm(fig, 9, 4, 100, 54.9)
lat = 24.5 + 0.1 * (np.arange(g01.shape[0]) + 0.5); lon = -125 + 0.1 * (np.arange(g01.shape[1]) + 0.5)
a.set_facecolor('white')
bg = a.pcolormesh(lon, lat, np.clip(g01, 0.1, 0.899), cmap=cm_bg, norm=nb_bg, shading='nearest', rasterized=True, zorder=0)
from matplotlib.path import Path
from matplotlib.patches import PathPatch
rings = [r for r in __import__('cstyle')._rings('ne_50m_land') if (r[:, 0].max() > -126) and (r[:, 0].min() < -66) and (r[:, 1].max() > 24) and (r[:, 1].min() < 50)]
vv = np.concatenate(rings); cc = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY] for r in rings])
clip = PathPatch(Path(vv, cc), transform=a.transData, fc='none', ec=LAND_EDGE, lw=0.3, zorder=2); a.add_patch(clip); bg.set_clip_path(clip)
from matplotlib.collections import LineCollection
a.add_collection(LineCollection(list(__import__('cstyle')._rings('ne_50m_states')), colors='white', linewidths=0.3, alpha=0.7, zorder=1))
p = s.groupby('site_no').agg(lat=('lat', 'first'), lon=('lon', 'first'), A=('drain_area_km2', 'first')).sort_values('A', ascending=False); p['ab'] = aclass(p.A).astype(str)
a.scatter(p.lon, p.lat, c=[ACOL_D[k] for k in p.ab], s=2.6, lw=0.12, edgecolors='#202020', rasterized=True, zorder=3)
a.set_xlim(-125, -66.5); a.set_ylim(24.5, 49.8); a.set_aspect(1 / np.cos(np.deg2rad(38)), adjustable='box')
geo_ticks(a, [-120, -110, -100, -90, -80, -70], [30, 35, 40, 45])
north(a, 0.935, 0.42); scalebar(a, -123.5, 26.3, 500, lat_ref=30)
stat(a, f'{len(s):,} crests\n{s.site_no.nunique():,} gauges', 0.985, 0.03, ha='right', va='bottom', fs=6.4, linespacing=1.0, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))

# ---------------- b: drainage areas
b = ax_mm(fig, 124, 4, 43, 20)
bins = np.logspace(0, 6.3, 43); xc_ = np.sqrt(bins[1:] * bins[:-1])
hu, _ = np.histogram(s.drain_area_km2, bins); hc, _ = np.histogram(C.area_km2, bins)
cls = np.digitize(xc_, BINS[1:-1])
b.bar(bins[:-1], hu, width=np.diff(bins), align='edge', color=[ACOL[min(k, 6)] for k in cls], lw=0, label=f'United States')
b.step(bins, np.r_[hc, hc[-1]], where='post', color=INK, lw=0.9, label=f'Six regions')
for x in BINS[1:-1]: b.axvline(x, color=GREY, lw=0.4, ls=':', zorder=0)
b.set_xscale('log'); b.set_xlim(1, 2e6); b.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6]); b.set_xticklabels(['1', '', '10$^2$', '', '10$^4$', '', '10$^6$']); b.xaxis.set_minor_locator(NullLocator())
b.set_xlabel('Drainage area (km$^2$)'); b.set_ylabel('Crests'); b.set_yticks([0, 2000, 4000]); b.set_yticklabels(['0', '2k', '4k'])
b.legend(handles=[Patch(fc=ACOL[3], lw=0), Line2D([], [], color=INK, lw=0.9)], labels=['United States', 'Six regions'], loc='upper right', fontsize=5.9, handlelength=1.0, bbox_to_anchor=(1.04, 1.10), frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, borderpad=0.25)

# ---------------- c: crests by dataset, storm-driven or not
c = ax_mm(fig, 141, 34.5, 26, 21.5)
rows = [('United States', len(s), np.nan, s.site_no.nunique(), INK)] + [(REGN[k], (C.dataset == k).sum(), (C[C.dataset == k].rain).mean(), C[C.dataset == k].basin_id.nunique(), REG[k]) for k in DS]
y = np.arange(len(rows))[::-1]
for yy, (nm, n, fr, g_, col) in zip(y, rows):
    if np.isnan(fr): c.barh(yy, n, color=col, height=0.7, lw=0)
    else: c.barh(yy, n * fr, color=col, height=0.7, lw=0); c.barh(yy, n * (1 - fr), left=n * fr, color=col, alpha=0.32, height=0.7, lw=0)
    c.text(n + 600, yy, f'{n / 1000:.1f}k' if n < 1e4 else f'{n / 1000:.0f}k', va='center', ha='left', fontsize=5.8, color=INK2)
c.set_yticks(y); c.set_yticklabels([r[0] for r in rows], fontsize=6.2); c.tick_params(axis='y', length=0)
c.set_xlim(0, 44000); c.set_xticks([0, 20000, 40000]); c.set_xticklabels(['0', '20k', '40k']); c.set_xlabel('Crests'); c.set_ylim(-0.6, len(rows) - 0.4)
spines(c, ('bottom',))
c.legend(handles=[Patch(fc=GREY, lw=0), Patch(fc=GREY, alpha=0.32, lw=0)], labels=['Storm-driven', 'Other'], loc='lower right', fontsize=5.8, handlelength=1.0, bbox_to_anchor=(1.08, 0.1))

# ---------------- d: Caravan gauges over the MODIS clear-sky climatology
T2 = 68; d = ax_mm(fig, 9, T2, 136, 53.4)
d.set_facecolor('white')
latg = 89.5 - np.arange(180); long = -179.5 + np.arange(360)
d.pcolormesh(long, latg, np.clip(m1, 0.1, 0.899), cmap=cm_bg, norm=nb_bg, shading='nearest', rasterized=True, zorder=0)
from matplotlib.collections import PolyCollection
d.add_collection(PolyCollection(__import__('cstyle')._rings('ne_50m_land'), facecolor='none', edgecolor='#6A6A6A', linewidth=0.25, zorder=1))
bas = C.drop_duplicates('basin_id')
for ds in DS:
    q = bas[bas.dataset == ds]; d.scatter(q.lon, q.lat, s=1.5, color=REG[ds], lw=0.1, edgecolors='#202020', rasterized=True, zorder=3)
d.set_xlim(-168, 158); d.set_ylim(-56, 72); d.set_aspect(1.0, adjustable='box')
geo_ticks(d, [-150, -100, -50, 0, 50, 100, 150], [-40, -20, 0, 20, 40, 60])
for k, (x, y_, ha) in {'namerica': (-70, 31, 'left'), 'camelsgb': (-12, 62, 'right'), 'lamah': (24, 53, 'left'), 'camelsbr': (-30, -12, 'left'),
                       'camelscl': (-66, -52, 'left'), 'camelsaus': (123, -43.5, 'center')}.items():
    d.text(x, y_, REGN[k], color=REG[k], fontsize=6.4, fontweight='bold', ha=ha, va='center', path_effects=HALO, zorder=6)
north(d, 0.04, 0.26, size=0.08); scalebar(d, -150, -48, 2000, lat_ref=-48, label='2,000 km')
stat(d, f'{len(C):,} crests, {C.basin_id.nunique():,} gauges', 0.985, 0.03, ha='right', va='bottom', fs=6.4, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))
sm = plt.cm.ScalarMappable(cmap=cm_bg, norm=nb_bg)
cb = cbar(fig, 151, T2 + 8, 2.0, 38, sm, orient='vertical', ticks=[0.1, 0.3, 0.5, 0.7, 0.9], fs=6.2)
fig.text(147.5 / W, 1 - (T2 + 1.5) / H, 'Mean clear-sky\nfraction, a and d', ha='left', va='top', fontsize=6.2, linespacing=0.95)

for l_, x_, t_ in [('a', 1, 2), ('b', 113, 2), ('c', 113, 32.5), ('d', 1, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig01_data')
print(len(s), s.site_no.nunique(), len(C), C.basin_id.nunique(), C.rain.sum(), C[C.rain].basin_id.nunique())
