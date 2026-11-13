# Fig. 1  Data and study domain
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
sm = plt.cm.ScalarMappable(cmap=cm_bg, norm=nb_bg)
FSL = 6.2                                                          # legend font size (pt)
RIGHT = 167.0                                                      # right edge shared by b, c and the legend column of d (mm)

def key(fig, x, t, handles, labels, title):
    """Marker legend with its top-left corner at (x, t) mm. Returns its bottom edge (mm from the top)."""
    lg = fig.legend(handles=handles, labels=labels, title=title, loc='upper left', bbox_to_anchor=(x / W, 1 - t / fig._H), fontsize=FSL, title_fontsize=FSL,
                    handlelength=0.7, handletextpad=0.3, labelspacing=0.22, borderpad=0, borderaxespad=0, frameon=False, alignment='left')
    fig.canvas.draw(); bb = lg.get_window_extent(fig.canvas.get_renderer())
    return (1 - bb.y0 / fig.bbox.height) * fig._H, bb.x1 / fig.bbox.width * W

def sky_bar(fig, x, t0, t1):
    """Clear-sky colour bar in a legend column: a two-line title with its top at t0 and the bar down to t1 (mm from the top)."""
    fig.text(x / W, 1 - t0 / fig._H, 'Clear-sky\nfraction', ha='left', va='top', fontsize=FSL, linespacing=1.0)
    tb = t0 + 5.6
    return cbar(fig, x + 0.2, tb, 1.7, t1 - tb, sm, orient='vertical', ticks=[0.1, 0.3, 0.5, 0.7, 0.9], fs=FSL)

def dot(c): return Line2D([], [], marker='o', ls='', ms=3.0, mfc=c, mec='#202020', mew=0.25)

H = 127.5; fig = figure(H)
AW = 92.9; AH = AW * 25.3 / 58.5 / np.cos(np.deg2rad(38))          # map a (mm); its legend column sits to the right
# ---------------- a: USGS gauges over the GOES-16 clear-sky climatology
a = ax_mm(fig, 9, 4, AW, AH)
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
# legend column of a: gauge colours, then the grey scale of the background
XA = 9 + AW + 1.4
ta, xa1 = key(fig, XA, 4.0, [dot(c_) for c_ in ACOL], LAB, 'Drainage\narea (km$^2$)')
sky_bar(fig, XA, ta + 3.6, 4 + AH - 0.9)

# ---------------- b: drainage areas
BL = 124; BH = 19.0
b = ax_mm(fig, BL, 4, RIGHT - BL, BH)
bins = np.logspace(0, 6.3, 43); xc_ = np.sqrt(bins[1:] * bins[:-1])
hu, _ = np.histogram(s.drain_area_km2, bins); hc, _ = np.histogram(C.area_km2, bins)
cls = np.digitize(xc_, BINS[1:-1])
b.bar(bins[:-1], hu, width=np.diff(bins), align='edge', color=[ACOL[min(k, 6)] for k in cls], lw=0, label=f'United States')
b.step(bins, np.r_[hc, hc[-1]], where='post', color=INK, lw=0.9, label=f'Six regions')
for x in BINS[1:-1]: b.axvline(x, color=GREY, lw=0.4, ls=':', zorder=0)
b.set_xscale('log'); b.set_xlim(1, 2e6); b.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6]); b.set_xticklabels(['1', '', '10$^2$', '', '10$^4$', '', '10$^6$']); b.xaxis.set_minor_locator(NullLocator())
b.set_xlabel('Drainage area (km$^2$)'); b.set_ylabel('Crests'); b.set_yticks([0, 2000, 4000]); b.set_yticklabels(['0', '2k', '4k'])
b.legend(handles=[Patch(fc=ACOL[3], lw=0), Line2D([], [], color=INK, lw=0.9)], labels=['United States', 'Six regions'], loc='upper right', fontsize=5.9, handlelength=1.0, bbox_to_anchor=(1.04, 1.10), frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, borderpad=0.25)

# ---------------- c: crests by dataset, storm-driven or not (its labels start where the axis title of b starts)
fig.canvas.draw(); R_ = fig.canvas.get_renderer()
b_left = b.yaxis.label.get_window_extent(R_).x0 / fig.bbox.width * W      # left edge of the 'Crests' title of b (mm)
CT = 32.3; CB = 4 + AH - 3.0
c = ax_mm(fig, 140, CT, RIGHT - 140, CB - CT)
rows = [('United States', len(s), np.nan, s.site_no.nunique(), INK)] + [(REGN[k], (C.dataset == k).sum(), (C[C.dataset == k].rain).mean(), C[C.dataset == k].basin_id.nunique(), REG[k]) for k in DS]
y = np.arange(len(rows))[::-1]
for yy, (nm, n, fr, g_, col) in zip(y, rows):
    if np.isnan(fr): c.barh(yy, n, color=col, height=0.7, lw=0)
    else: c.barh(yy, n * fr, color=col, height=0.7, lw=0); c.barh(yy, n * (1 - fr), left=n * fr, color=col, alpha=0.32, height=0.7, lw=0)
    c.text(n + 600, yy, f'{n / 1000:.1f}k' if n < 1e4 else f'{n / 1000:.0f}k', va='center', ha='left', fontsize=5.8, color=INK2)
c.set_yticks(y); c.set_yticklabels([r[0] for r in rows], fontsize=6.2); c.tick_params(axis='y', length=0)
c.set_xlim(0, 44000); c.set_xticks([0, 20000, 40000]); c.set_xticklabels(['0', '20k', '40k']); c.set_xlabel('Crests'); c.set_ylim(-0.6, len(rows) - 0.4)
spines(c, ('bottom',))
fig.canvas.draw()
c_left = min(t_.get_window_extent(R_).x0 for t_ in c.get_yticklabels()) / fig.bbox.width * W
CL = 140 - (c_left - b_left)                                            # move the axis so that the widest label starts at b_left
low = lambda arts: (1 - min(t_.get_window_extent(R_).y0 for t_ in arts) / fig.bbox.height) * H
CB -= low([c.xaxis.label]) - low(a.get_xticklabels())                   # the axis title of c ends level with the longitude labels of a
c.set_position([CL / W, 1 - CB / H, (RIGHT - CL) / W, (CB - CT) / H])
c.legend(handles=[Patch(fc=GREY, lw=0), Patch(fc=GREY, alpha=0.32, lw=0)], labels=['Storm-driven', 'Other'], loc='lower right', fontsize=5.8, handlelength=1.0, bbox_to_anchor=(1.03, 0.1))

# ---------------- d: Caravan gauges over the MODIS clear-sky climatology
T2 = 66; DW = 139.0; DH = DW * 128 / 326
d = ax_mm(fig, 9, T2, DW, DH)
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
north(d, 0.04, 0.26, size=0.08); scalebar(d, -150, -48, 2000, lat_ref=-48, label='2,000 km')
# legend column of d, built like that of a
XD = 9 + DW + 1.4
td, xd1 = key(fig, XD, T2, [dot(REG[k]) for k in DS], [REGN[k] for k in DS], 'Region')
sky_bar(fig, XD, td + 3.6, T2 + DH - 0.9)

for l_, x_, t_ in [('a', 1, 2), ('b', b_left - 0.3, 2), ('c', b_left - 0.3, CT - 3.4), ('d', 1, T2 - 3)]: letter(fig, l_, x_, t_)
print('legend columns end at', round(xa1, 1), round(xd1, 1), 'mm; title of b starts at', round(b_left, 1), '; axis of c from', round(CL, 1))
save(fig, 'Fig01_test' if 'test' in sys.argv else 'Fig01_data')
print(len(s), s.site_no.nunique(), len(C), C.basin_id.nunique(), C.rain.sum(), C[C.rain].basin_id.nunique())
