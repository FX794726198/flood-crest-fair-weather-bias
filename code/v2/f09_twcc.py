# Fig. 9  The timing-weighted crest correction (TWCC): loss over the wait, the weight law and its skill
from cstyle import *
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
F = 0.3048
M = pd.read_parquet('data/rev_recon.parquet'); M['ab'] = aclass(M.A).astype(str)
S = pd.read_parquet('data/rev_recon_storm.parquet'); S['ab'] = aclass(S.A).astype(str)
V1 = pd.read_parquet('data/recon_cv.parquet'); EX = pd.read_parquet('data/v3_exposure.parquet'); Z = np.load('data/recon_map.npz')
R = pd.read_parquet('data/v3_twcc.parquet'); RS = pd.read_parquet('data/v3_twcc_storm.parquet'); LAW = np.load('data/v3_twcc_laws.npy').mean(0)
TW = '#1B7F5C'                                                                                   # colour of the corrected crest
def w25(d, est, obs='dpk'): return (np.abs(d[est] / d[obs] - 1) <= 0.25).mean()
H = 120; fig = figure(H); T1, H1 = 4, 38

# ---------------- a: crest depth / first-look depth against the wait
a = ax_mm(fig, 12, T1, 40, H1)
for i, k in enumerate(LABK):
    d = M[M.ab == k]; g = d.groupby('lag').y; t = g.median(); n = g.size(); t = t[(n >= 25) & (t.index <= 6)]
    a.plot(t.index, np.exp(t.values), '-', color=ACOL[i], lw=1.2, zorder=3); a.plot(t.index, np.exp(t.values), 'o', color=ACOL[i], ms=2.7, mec='white', mew=0.35, zorder=4)
a.set_yscale('log'); a.set_yticks([1, 2, 5, 10]); a.set_yticklabels(['1', '2', '5', '10']); a.yaxis.set_minor_locator(NullLocator()); a.set_ylim(0.9, 24)
a.set_xlim(-0.3, 6.3); a.set_xticks(range(7)); a.set_xlabel('Days from crest to first clear look'); a.set_ylabel('Crest depth / first-look depth')
a.legend([Line2D([], [], color=c, lw=1.4) for c in ACOL], LAB, title='Area (km$^2$)', title_fontsize=6.0, fontsize=5.9, loc='upper right', ncol=2,
         columnspacing=0.6, handlelength=1.0, labelspacing=0.15, bbox_to_anchor=(1.03, 1.04))

# ---------------- b: weight of the image against the wait, cells (dots) and the fitted law (lines)
b = ax_mm(fig, 66, T1, 38, H1)
ri = np.log(R.dpk / R.e_img); rp = np.log(R.dpk / R.e_pri); R['tb'] = np.clip(np.floor(R.lag), 0, 4)
G = pd.DataFrame({'vi': ri ** 2, 'vp': rp ** 2, 'c': ri * rp, 'ab': R.ab, 'tb': R.tb}).groupby(['ab', 'tb']).agg(vi=('vi', 'mean'), vp=('vp', 'mean'), c=('c', 'mean'), n=('vi', 'size')).reset_index()
G['w'] = np.clip((G.vp - G.c) / (G.vi + G.vp - 2 * G.c), 0, 1)
tt = np.linspace(0, 4.3, 100); kexp = -LAW[2]; Aexp = LAW[1] / np.log(10) / kexp; Tco = np.exp(LAW[0] / kexp)
for i, k in enumerate(LABK):
    g = G[(G.ab == k) & (G.n >= 25)]
    b.plot(tt, 1 / (1 + np.exp(-(LAW[0] + LAW[1] * np.log10(XC[i]) + LAW[2] * np.log(tt + 1)))), '-', color=ACOL[i], lw=1.1, zorder=3)
    b.plot(g.tb, g.w, 'o', color=ACOL[i], ms=2.7, mec='white', mew=0.35, zorder=4)
b.set_xlim(-0.25, 4.3); b.set_xticks(range(5)); b.set_xticklabels(['0', '1', '2', '3', '≥4']); b.set_ylim(0, 1.02); b.set_yticks([0, 0.25, 0.5, 0.75, 1])
b.set_xlabel('Days from crest to first clear look, τ'); b.set_ylabel('Weight of the image, w')
b.axhline(0.5, color=GREY_L, lw=0.5, ls=(0, (2, 2)), zorder=1)
b.text(0.03, 0.012, f'w = 1 / [1 + ((τ + 1) / T$_w$)$^{{{kexp:.2f}}}$]\nT$_w$ = {Tco:.2f} A$^{{{Aexp:.2f}}}$ days', transform=b.transAxes, ha='left', va='bottom', fontsize=5.6,
       color=INK, linespacing=1.2)

# ---------------- c: share within +-25% by area class for the first look, the two branches and the correction
c = ax_mm(fig, 117, T1, 50, H1)
EST = [('e_look', 'First clear look', FIRST, '-', 'o', R, 1.0), ('e_img', 'Image branch', '#8C8C8C', '-', 's', R, 1.0),
       ('e_pri', 'Prior branch (no image)', '#4D4D4D', ':', '^', R, 1.0), ('e_law', 'TWCC', TW, '-', 'o', R, 1.7),
       ('e_law', 'TWCC, radar storm day', '#66C2A4', '--', 'o', RS, 1.1)]
for est, nm, col, ls, mk, D, lw in EST:
    y = [w25(D[D.ab == k], est) for k in LABK]
    c.plot(XC, y, ls, color=col, lw=lw, label=nm, zorder=4 if lw > 1.5 else 3); c.plot(XC, y, mk, color=col, ms=2.6, mec='white', mew=0.3, zorder=5)
area_axis(c); c.set_ylim(0, 1.13); c.set_yticks([0, 0.25, 0.5, 0.75, 1]); c.set_ylabel('Crests within ±25%')
c.legend(loc='upper center', fontsize=5.7, handlelength=1.4, bbox_to_anchor=(0.5, 1.04), labelspacing=0.15, ncol=2, columnspacing=0.7)

# ---------------- d-f: mapped examples within 2.5 km of the gauge
CS = {'seen': '#3B7DBF', 'rec': '#1B7F5C', 'miss': '#D95F02', 'over': '#F2C14E', 'chan': '#0B2A4A'}
CM4 = ListedColormap([CS['seen'], CS['rec'], CS['miss'], CS['over']])
def mapit(l, t, w, tag, site, pdate, title, sb=1, R=2.5):
    a = ax_mm(fig, l, t, w, w)
    r = V1[(V1.site_no == site) & (V1.peak_date == pdate)].iloc[0]; x = EX[(EX.site_no == site) & (EX.peak_date == pdate)].iloc[0]
    h = Z[f'{tag}_hand'].astype(float); tr = Z[f'{tag}_ht']; p = Z[f'{tag}_pop'].astype(float); ny, nx = h.shape
    lon = tr[2] + (np.arange(nx) + 0.5) * tr[0]; lat = tr[5] + (np.arange(ny) + 0.5) * tr[4]
    lat0, lon0 = r.lat, r.lon; dc = r.dh_peak_ft * F; df = r.dh_first_ft * F; dr = x.e_law * F; lo_ = min(dc, dr); v = h > -1000
    cls = np.full(h.shape, np.nan)
    cls[v & (h > 0.01) & (h <= df)] = 0; cls[v & (h > df) & (h <= lo_)] = 1; cls[v & (h > lo_) & (h <= dc) & (dr < dc)] = 2; cls[v & (h > dc) & (h <= dr)] = 3
    a.set_facecolor('#F7F7F7')
    a.pcolormesh(lon, lat, np.where(p > 0, p, np.nan), cmap='Greys', norm=LogNorm(0.3, 30), shading='nearest', rasterized=True, alpha=0.6)
    a.pcolormesh(lon, lat, cls, cmap=CM4, vmin=-0.5, vmax=3.5, shading='nearest', alpha=0.9, rasterized=True)
    a.pcolormesh(lon, lat, np.where(v & (h <= 0.01), 1, np.nan), cmap=ListedColormap([CS['chan']]), shading='nearest', rasterized=True)
    a.plot(lon0, lat0, '^', ms=6, mfc='#FFD92F', mec='k', mew=0.6, zorder=5)
    k = 1 / np.cos(np.deg2rad(lat0)); dl = R / 111.32; a.set_aspect(k, adjustable='datalim'); a.set_xlim(lon0 - dl * k, lon0 + dl * k); a.set_ylim(lat0 - dl, lat0 + dl)
    xt = np.round([lon0 - 0.55 * dl * k, lon0 + 0.55 * dl * k], 2); yt = np.round([lat0 - 0.55 * dl, lat0 + 0.55 * dl], 2)
    a.set_xticks(xt); a.set_xticklabels([f'{abs(v_):.2f}°W' for v_ in xt], fontsize=6.0); a.set_yticks(yt)
    a.set_yticklabels([f'{v_:.2f}°N' for v_ in yt], fontsize=6.0, rotation=90, va='center'); a.tick_params(length=2, pad=1); frame(a)
    xr = a.get_xlim(); yr = a.get_ylim(); kd = 111.32 * np.cos(np.deg2rad(lat0)); x0 = xr[0] + 0.07 * (xr[1] - xr[0]); y0 = yr[0] + 0.07 * (yr[1] - yr[0])
    a.add_patch(plt.Rectangle((x0 - 0.03 * (xr[1] - xr[0]), y0 - 0.035 * (yr[1] - yr[0])), sb / kd + 0.06 * (xr[1] - xr[0]), 0.13 * (yr[1] - yr[0]), fc='white', ec='none', alpha=0.85, zorder=6))
    a.plot([x0, x0 + sb / kd], [y0, y0], 'k-', lw=1.8, solid_capstyle='butt', zorder=7); a.text(x0 + sb / kd / 2, y0 + 0.022 * (yr[1] - yr[0]), f'{sb} km', ha='center', va='bottom', fontsize=6.0, zorder=7)
    north(a, 0.92, 0.80, size=0.09)
    a.set_title(title, fontsize=6.8, pad=2.5)
    a.text(0.5, -0.1, f'Crest {dc:.1f} m, first look {df:.1f} m, TWCC {dr:.1f} m\nResidents seen {x.P_first/x.P_crest:.0%}, after correction {x.P_tw/x.P_crest:.0%}',
           transform=a.transAxes, fontsize=6.1, va='top', ha='center', linespacing=1.1)
    return a
T2 = 61; WM = 44
mapit(10, T2, WM, 'nc', '02124269', '2020-11-12', 'Charlotte, NC, 19 km$^2$, Nov 2020')
mapit(66, T2, WM, 'ca', '11466170', '2021-10-24', 'Santa Rosa, CA, 54 km$^2$, Oct 2021')
mapit(122, T2, WM, 'ia', '05464500', '2019-03-19', 'Cedar Rapids, IA, 16,861 km$^2$, Mar 2019')
fig.legend(handles=[Patch(fc=CS['seen']), Patch(fc=CS['rec']), Patch(fc=CS['miss']), Patch(fc=CS['over']), Patch(fc=CS['chan']), Patch(fc='#9E9E9E'),
                    Line2D([], [], marker='^', ls='', mfc='#FFD92F', mec='k', ms=5)],
           labels=['Seen at first clear look', 'Recovered by TWCC', 'Still missed', 'Corrected beyond crest', 'Drainage network', 'Residents', 'Gauge'],
           loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.0), handlelength=1.1, columnspacing=0.8, fontsize=6.1, handletextpad=0.35)
for l_, x_, t_ in [('a', 1, 2), ('b', 55, 2), ('c', 106, 2), ('d', 1, T2 - 5), ('e', 57, T2 - 5), ('f', 113, T2 - 5)]: letter(fig, l_, x_, t_)
save(fig, 'Fig09_reconstruction')
