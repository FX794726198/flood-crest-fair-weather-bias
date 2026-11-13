# Fig. 4  Crests of small basins hide under storm cloud in the United States
from cstyle import *
from matplotlib.colors import BoundaryNorm, ListedColormap
s = pd.read_parquet('data/main_sample.parquet'); s['ab'] = aclass(s.drain_area_km2)
R = pd.read_csv('fig/fig1_bins.csv')                    # crest-day ratio, near-crest ratio, 95% CIs, flow quantiles (300 bootstraps)
ST = pd.read_parquet('data/stage_first.parquet'); EX = pd.read_parquet('data/extent_first.parquet')
for X in (ST, EX): X['ab'] = aclass(X.A)

H = 121; fig = figure(H)
# ---------------- a: map of the flow left at the first clear look
a = ax_mm(fig, 9, 4, 101, 55.5)
basemap(a, (-125, -66.5, 24.5, 49.8), states=True); a.set_aspect(1 / np.cos(np.deg2rad(38)), adjustable='box')
p = s.groupby('site_no').agg(lat=('lat', 'first'), lon=('lon', 'first'), q=('q_at_first', 'median'), n=('q_at_first', 'size'))
p = p[p.n >= 2].sort_values('q', ascending=False)
cm = ListedColormap(plt.get_cmap('RdYlBu')(np.linspace(0.02, 0.98, 10))); nb = BoundaryNorm(np.linspace(0, 1, 11), 10)
sc = a.scatter(p.lon, p.lat, c=p.q, s=3.2, cmap=cm, norm=nb, lw=0.15, edgecolors='#404040', rasterized=True, zorder=3)
geo_ticks(a, [-120, -110, -100, -90, -80, -70], [30, 35, 40, 45])
north(a, 0.935, 0.42); scalebar(a, -123.5, 26.3, 500, lat_ref=30)
cb = cbar(fig, 91, 49.5, 16, 1.6, sc, ticks=[0, 0.5, 1], fs=6.2); cb.ax.set_xticklabels(['0', '0.5', '1'])
fig.text(99 / W, 1 - 47.6 / H, 'Flow left at first\nclear look (median)', ha='center', va='bottom', fontsize=6.2, linespacing=0.95)

# ---------------- b: clear sky around the crest by size
b = ax_mm(fig, 125, 4, 42, 49)
OFF = list(range(-3, 11)); groups = [('<100 km$^2$', s.drain_area_km2 < 100, ACOL[0]), ('100–1,000', s.drain_area_km2.between(100, 1000), ACOL[2]),
                                      ('1,000–10,000', s.drain_area_km2.between(1000, 1e4), ACOL[4]), ('>10,000', s.drain_area_km2 > 1e4, ACOL[6])]
b.axvspan(-0.5, 0.5, color=GREY_LL, lw=0, zorder=0)
for nm, sel, c in groups:
    y = [np.nanmean(s.loc[sel, f'b{o}']) / np.nanmean(s.loc[sel, 'clim_b']) for o in OFF]
    b.plot(OFF, y, '-', color=c, lw=1.3, label=nm, zorder=3); b.plot(OFF, y, 'o', color=c, ms=2.6, mec='white', mew=0.4, zorder=4)
ref(b); b.set_xlim(-3.4, 10.4); b.set_ylim(0.3, 1.15); b.set_xticks([-3, 0, 3, 6, 9]); b.set_yticks([0.4, 0.6, 0.8, 1.0])
b.set_xlabel('Days from crest'); b.set_ylabel('Clear sky / monthly normal')
b.legend(loc='lower right', fontsize=6.3, handlelength=1.2, bbox_to_anchor=(1.02, 0.0))
b.text(0, 1.13, 'Crest day', ha='center', va='top', fontsize=FSS, color=INK2)

# ---------------- c: crest-day ratio by size
T2 = 71; HH = 38
c = ax_mm(fig, 13, T2, 40, HH)
band(c, XC, R.rcl, R.rch, '#6E9CC7', 0.22); c.plot(XC, R.rc, '-', color='#2F6DA8', lw=1.3, label='Crest day ±1 day')
band(c, XC, R.r0l, R.r0h, OBS, 0.2); c.plot(XC, R.r0, '-', color=OBS, lw=1.3, label='Crest day')
for i in range(7): c.plot(XC[i], R.r0[i], 'o', color=ACOL[i], ms=4.2, mec='white', mew=0.5, zorder=5)
ref(c); area_axis(c); c.set_ylim(0.3, 1.22); c.set_yticks([0.4, 0.6, 0.8, 1.0]); c.set_ylabel('Clear sky / monthly normal')
c.legend(loc='upper left', fontsize=6.3, bbox_to_anchor=(-0.02, 1.04), ncol=1)
c.text(XC[0], R.r0[0] - 0.05, f'{R.r0[0]:.2f}', ha='center', va='top', fontsize=FSS, color=OBS)
c.text(XC[6], R.r0[6] - 0.05, f'{R.r0[6]:.2f}', ha='center', va='top', fontsize=FSS, color=OBS)

# ---------------- d: what the first clear look keeps
d = ax_mm(fig, 70, T2, 40, HH)
band(d, XC, R.q25, R.q75, GREY, 0.18)
d.plot(XC, R.q50, '-o', color=INK, lw=1.3, ms=3, mec='white', mew=0.4, label='Discharge')
d.plot(XC, [ST[ST.ab == k].depth_frac.median() for k in LABK], '-s', color=WATER, lw=1.3, ms=2.8, mec='white', mew=0.4, label='Water depth')
d.plot(XC, [EX[EX.ab == k].extent_frac.median() for k in LABK], '-^', color='#2E8B57', lw=1.3, ms=3, mec='white', mew=0.4, label='Flooded area')
area_axis(d); d.set_ylim(0, 1.05); d.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); d.set_ylabel('Left at first clear look / crest')
d.legend(loc='lower right', fontsize=6.3, bbox_to_anchor=(1.03, 0.0))
d.text(XC[0], R.q50[0] + 0.07, f'{R.q50[0]*100:.0f}%', ha='center', va='bottom', fontsize=FSS, color=INK)

# ---------------- e: flashiness
e = ax_mm(fig, 125, T2, 42, HH)
s['fl'] = pd.qcut(s.inst_to_daily, 8)
g = s.groupby('fl', observed=True); xm = g.inst_to_daily.median(); yr = g.b0.mean() / g.clim_b.mean(); yq = g.q_at_first.median()
e.plot(xm, yr, '-o', color=OBS, lw=1.3, ms=3, mec='white', mew=0.4, label='Crest-day clear sky / normal')
e.plot(xm, yq, '-o', color=INK, lw=1.3, ms=3, mec='white', mew=0.4, label='Flow at first clear look / crest')
e.set_xscale('log'); e.set_xticks([1, 2, 4, 8]); e.set_xticklabels(['1', '2', '4', '8']); e.xaxis.set_minor_locator(NullLocator())
e.set_xlim(0.92, 9); e.set_ylim(0, 1.05); e.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
e.set_xlabel('Instantaneous / daily-mean crest'); e.set_ylabel('Ratio')
e.legend(loc='upper right', fontsize=6.3, bbox_to_anchor=(1.03, 1.03))

for l_, x_, t_ in [('a', 1, 2), ('b', 114, 2), ('c', 1, T2 - 3), ('d', 58, T2 - 3), ('e', 114, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig04_us_phenomenon')
