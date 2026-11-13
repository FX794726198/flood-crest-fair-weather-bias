# Fig. 3  Two floods: GOES-16 clear/cloud at the Aqua overpass on the crest day and the two days after, with the 15-minute stage record
from cstyle import *
import datetime as dt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
Z = np.load('data/case_goes.npz')
TT = np.load('data/terra.npy', mmap_mode='r'); AA = np.load('data/aqua.npy', mmap_mode='r'); D = np.load('data/days.npy', allow_pickle=True); di = {str(d)[:10]: i for i, d in enumerate(D)}
v = pd.read_parquet('data/usgs/iv.parquet'); v = v[v.param == '00065'].dropna(subset=['value']); ev = pd.read_csv('data/usgs/events.csv', dtype={'site_no': str})
_cs = __import__('cstyle')
SL = _cs._rings('ne_50m_states')
RV = []
for f in json.load(open('data/ne_50m_rivers.geojson'))['features']:
    g = f['geometry']
    if g is None: continue
    for p in (g['coordinates'] if g['type'] == 'MultiLineString' else [g['coordinates']]): RV.append(np.array(p))
CLR = '#4F8FB0'; CLD = '#FFFFFF'; MISS = '#D9D9D9'
cm = ListedColormap([MISS, CLR, CLD])
CASES = [('nc', '02124269', '2020-11-12', 35.309, -80.674, 19.3, 'Little Sugar Creek, Charlotte, North Carolina, 19 km$^2$', 1.6, 50),
         ('sd', '06478500', '2024-06-23', 43.186, -97.635, 53540, 'James River, South Dakota, 53,540 km$^2$', 3.2, 100)]
H = 104; fig = figure(H)
MW = 29.0; L0 = 12; GAP = 1.6; HL = 121; HW = 46
rowT = [9, 57]
let = iter('abcd')
for row, (tag, site, pdate, lat0, lon0, area, title, dd, sb) in enumerate(CASES):
    T0 = rowT[row]; LA = Z[f'{tag}_lat']; LO = Z[f'{tag}_lon']
    days = [(dt.date.fromisoformat(pdate) + dt.timedelta(k)).isoformat() for k in (0, 1, 2)]
    asp = 1 / np.cos(np.deg2rad(lat0)); mh = MW * (2 * dd) * asp / (2 * dd * 1.25)
    for j, day in enumerate(days):
        a = ax_mm(fig, L0 + j * (MW + GAP), T0, MW, mh)
        b_ = Z[f'{tag}_{day}']; img = np.where(b_ < 0, 0, np.where(b_ == 0, 1, 2))
        a.pcolormesh(LO, LA, img, cmap=cm, vmin=0, vmax=2, shading='nearest', rasterized=True, zorder=0)
        a.add_collection(LineCollection(SL, colors='#7F7F7F', linewidths=0.4, zorder=1))
        a.add_collection(LineCollection(RV, colors='#1F5F8B', linewidths=0.6, zorder=1))
        R = max(np.sqrt(area / np.pi) / 111.32, 0.06)
        a.add_patch(Ellipse((lon0, lat0), 2 * R * asp, 2 * R, fill=False, ec=OBS, lw=1.0, ls=(0, (3, 1.5)), zorder=3))
        a.plot(lon0, lat0, marker='^', ms=5.5, mfc='#FFD92F', mec='k', mew=0.6, zorder=4)
        a.set_xlim(lon0 - dd * 1.25, lon0 + dd * 1.25); a.set_ylim(lat0 - dd, lat0 + dd); a.set_aspect(asp, adjustable='box')
        k = di.get(day); iy = int((lat0 - 24.5) / 0.1); ix = int((lon0 + 125) / 0.1); fA = AA[k, iy, ix] / 200
        a.set_title(['Crest day', 'Day 1', 'Day 2'][j], fontsize=7.0, pad=2)
        a.text(0.04, 0.05, 'Clear at gauge' if fA >= 0.5 else 'Cloud at gauge', transform=a.transAxes, fontsize=6.0, color='white' if fA >= 0.5 else INK, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.25', fc=CLR if fA >= 0.5 else 'white', ec='#7F7F7F', lw=0.4), zorder=6)
        st = 1 if tag == 'nc' else 2
        xt = np.arange(np.ceil(lon0 - dd * 1.25), lon0 + dd * 1.25, st); yt = np.arange(np.ceil(lat0 - dd), lat0 + dd, st)
        a.set_xticks(xt); a.set_xticklabels([f'{abs(x):.0f}°W' for x in xt], fontsize=6.0); a.set_yticks(yt)
        a.set_yticklabels([f'{y:.0f}°N' for y in yt] if j == 0 else [], fontsize=6.0, rotation=90, va='center')
        a.tick_params(length=2, pad=1); frame(a)
        if j == 0:
            xr = a.get_xlim(); yr = a.get_ylim(); kd = 111.32 * np.cos(np.deg2rad(lat0))
            x0 = xr[0] + 0.07 * (xr[1] - xr[0]); y0 = yr[1] - 0.12 * (yr[1] - yr[0])
            a.add_patch(plt.Rectangle((x0 - 0.03 * (xr[1] - xr[0]), y0 - 0.05 * (yr[1] - yr[0])), sb / kd + 0.06 * (xr[1] - xr[0]), 0.15 * (yr[1] - yr[0]), fc='white', ec='none', alpha=0.85, zorder=6))
            a.plot([x0, x0 + sb / kd], [y0, y0], 'k-', lw=1.8, solid_capstyle='butt', zorder=7)
            a.text(x0 + sb / kd / 2, y0 + 0.02 * (yr[1] - yr[0]), f'{sb} km', ha='center', va='bottom', fontsize=6.0, zorder=7)
        if j == 2:
            a.add_patch(plt.Rectangle((0.82, 0.62), 0.14, 0.33, transform=a.transAxes, fc='white', ec='none', alpha=0.85, zorder=2))
            north(a, 0.89, 0.75, size=0.15)
    fig.text(L0 / W, 1 - (T0 - 5.2) / H, title, fontsize=7.2, fontweight='bold', ha='left', va='bottom')
    letter(fig, next(let), 1, T0 - 6)
    # hydrograph with every Terra and Aqua overpass
    a = ax_mm(fig, HL, T0, HW, mh - 3.6)          # shorter than the maps, so its axis label ends level with their tick labels
    g = v[(v.site_no == site) & (v.peak_date == pdate)].sort_values('time_utc'); r = ev[(ev.site_no == site) & (ev.peak_date == pdate)].iloc[0]
    t = (g.time_utc - pd.Timestamp(pdate, tz='UTC')).dt.total_seconds().values / 3600; h = g.value.values
    ic = np.argmax(h); tc = t[ic]; pre = (t < tc) & (t >= tc - 72); hb = h[pre].min(); y = (h - hb) / (h[ic] - hb)
    a.fill_between((t - tc) / 24, 0, y, color=WATER_L, alpha=0.6, lw=0); a.plot((t - tc) / 24, y, color=WATER, lw=1.2)
    iy = int((r.lat - 24.5) / 0.1); ix = int((r.lon + 125) / 0.1); first = None
    for o in range(-2, 8):
        k = di.get((dt.date.fromisoformat(pdate) + dt.timedelta(o)).isoformat())
        for loc, cube in ((10.5, TT), (13.5, AA)):
            tu = o * 24 + loc - r.lon / 15
            if tu < t[0] or tu > t[-1] or k is None: continue
            f = cube[k, iy, ix] / 200; yy = np.interp(tu, t, y); x = (tu - tc) / 24
            a.plot(x, yy, 'o', ms=4.0, mfc=CLR if f >= 0.5 else 'white', mec=INK if f >= 0.5 else GREY, mew=0.6, zorder=4)
            if f >= 0.5 and x >= -0.05 and first is None: first = (x, yy)
    if first:
        a.annotate(f'First clear look,\n{first[1] * 100:.0f}% of crest depth', xy=first, xytext=(first[0] + 0.9, min(first[1] + 0.38, 0.86)) if first[1] < 0.5 else (first[0] + 0.7, 0.3), fontsize=6.2,
                   arrowprops=dict(arrowstyle='->', lw=0.6, color=INK), va='bottom', linespacing=1.0)
    a.axvline(0, color=GREY, lw=0.6, ls=(0, (3, 2))); a.set_xlim(-1.5, 5); a.set_ylim(-0.03, 1.08); a.set_xticks(range(-1, 6)); a.set_yticks([0, 0.5, 1])
    a.set_xlabel('Days from crest'); a.set_ylabel('Depth / crest depth')
    letter(fig, next(let), HL - 11, T0 - 6)
fig.legend(handles=[Patch(fc=CLR, ec='#7F7F7F', lw=0.3), Patch(fc=CLD, ec='#7F7F7F', lw=0.3), Patch(fc=MISS, ec='#7F7F7F', lw=0.3),
                    Line2D([], [], marker='^', ls='', mfc='#FFD92F', mec='k', ms=5), Line2D([], [], color=OBS, ls=(0, (3, 1.5)), lw=1.0),
                    Line2D([], [], marker='o', ls='', mfc=CLR, mec=INK, ms=4), Line2D([], [], marker='o', ls='', mfc='white', mec=GREY, ms=4)],
           labels=['Clear (GOES-16)', 'Cloud', 'No retrieval', 'Gauge', 'Basin-sized circle', 'Clear overpass', 'Cloudy overpass'],
           loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=0.9, fontsize=6.2, handletextpad=0.35)
save(fig, 'Fig03_case')
