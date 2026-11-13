# Fig. 11  What current and possible observing systems would capture of the crest (390 gauged 15-minute crests with a defined extent)
from cstyle import *
from scipy import stats
S = pd.read_parquet('data/osse_schemes.parquet'); Wp = pd.read_parquet('data/osse_sweep.parquet'); Du = pd.read_parquet('data/osse_duration.parquet')
for X in (S, Wp, Du): X['ab'] = X.ab.astype(str)
PCI = pd.read_csv('fig/rev/osse_people_ci.csv', index_col=0); RQ = pd.read_csv('fig/rev/osse_required_revisit_ci.csv', index_col=0)
CL = ACOL_D

H = 113; fig = figure(H); T1, H1 = 4, 36; T2, H2 = 58, 44
# ---------------- a: crest duration against drainage area, with the gaps between looks
a = ax_mm(fig, 12, T1, 41, H1)
d = Du[Du.D50 > 0].merge(pd.read_parquet('data/osse_edge.parquet'), on=['site_no', 'peak_date'], how='left')
full = ~(d.edge_start | d.edge_end)
a.scatter(d.A[full], d.D50[full], c=[CL[k] for k in d.ab[full]], s=4, lw=0, alpha=0.75, rasterized=True, zorder=2)
for sel, mk in [(d.edge_start & ~d.edge_end, '<'), (d.edge_end & ~d.edge_start, '>'), (d.edge_start & d.edge_end, 'D')]:   # durations that are lower bounds
    a.scatter(d.A[sel], d.D50[sel], marker=mk, s=9 if mk != 'D' else 7, facecolors='white', edgecolors=[CL[k] for k in d.ab[sel]], lw=0.6, zorder=3)
lr = stats.linregress(np.log10(d.A), np.log10(d.D50)); xx = np.logspace(0, 5.6, 50); a.plot(xx, 10 ** (lr.intercept + lr.slope * np.log10(xx)), '-', color=INK, lw=1.1, zorder=3)
med = d.groupby('ab').agg(A=('A', 'median'), m=('D50', 'median'), lo=('D50', lambda x: x.quantile(.25)), hi=('D50', lambda x: x.quantile(.75))).reindex(LABK)
for i, k in enumerate(LABK):
    a.plot([med.A[k]] * 2, [med.lo[k], med.hi[k]], '-', color=INK, lw=1.0, zorder=4)
    a.plot(med.A[k], med.m[k], 'o', color=ACOL[i], mec=INK, mew=0.6, ms=4.0, zorder=5)
for h_, nm in [(21, '21 h'), (144, '6 d'), (288, '12 d')]:      # revisit gaps, named in the caption and in panel d
    a.axhline(h_, color=PRED, lw=0.6, ls=(0, (4, 2)), zorder=1)
    a.text(1.02, h_, nm, transform=a.get_yaxis_transform(), fontsize=5.9, color=PRED, va='center', ha='left', clip_on=False)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(1, 4e5); a.set_ylim(0.3, 1500)
a.set_xticks([1, 10, 100, 1e3, 1e4, 1e5]); a.set_xticklabels(['1', '10', '10$^2$', '10$^3$', '10$^4$', '10$^5$']); a.xaxis.set_minor_locator(NullLocator())
a.set_yticks([1, 10, 100, 1000]); a.set_yticklabels(['1', '10', '100', '1,000']); a.yaxis.set_minor_locator(NullLocator())
a.set_xlabel('Drainage area (km$^2$)'); a.set_ylabel('Crest duration (h)')
print('11a slope, r', round(lr.slope, 3), round(lr.rvalue, 3))

# ---------------- b: captured extent against revisit, with the revisit that gives 80%
b = ax_mm(fig, 69, T1, 41, H1)
G = Wp.groupby(['ab', 'rv']).ext.mean().unstack(0)
for i, k in enumerate(LABK):
    b.plot(G.index, G[k], '-', color=ACOL[i], lw=1.3, label=LAB[i], zorder=3)
    b.errorbar(RQ.req_h[k], 0.8, xerr=[[RQ.req_h[k] - RQ.lo[k]], [RQ.hi[k] - RQ.req_h[k]]], fmt='o', color=ACOL[i], mec=INK, mew=0.5, ms=3.6, lw=0.9, zorder=5)
b.axhline(0.8, color=GREY, lw=0.6, ls=':', zorder=1)
b.set_xscale('log'); b.set_xticks([1, 6, 24, 72, 288]); b.set_xticklabels(['1 h', '6 h', '1 d', '3 d', '12 d']); b.xaxis.set_minor_locator(NullLocator())
b.set_xlim(0.9, 320); b.set_ylim(0, 1.03); b.set_yticks([0, 0.25, 0.5, 0.75, 1]); b.set_xlabel('Revisit interval, no cloud'); b.set_ylabel('Crest extent captured')
b.legend(title='Area (km$^2$)', title_fontsize=6.0, fontsize=5.8, loc='lower left', ncol=2, handlelength=1.0, columnspacing=0.6, labelspacing=0.15, bbox_to_anchor=(-0.02, -0.02))

# ---------------- c: one curve for all basins
c = ax_mm(fig, 126, T1, 41, H1)
X = Wp.merge(Du[['site_no', 'peak_date', 'D50']], on=['site_no', 'peak_date']); X = X[X.D50 > 0]; X['x'] = X.D50 / X.rv
bins = np.logspace(-2.5, 2.5, 26); xc = np.sqrt(bins[1:] * bins[:-1]); X['xb'] = pd.cut(X.x, bins)
for i, k in enumerate(LABK):
    t = X[X.ab == k].groupby('xb', observed=False).ext.agg(['mean', 'size']); m = (t['size'] >= 20).values
    c.plot(xc[m], t['mean'].values[m], '-', color=ACOL[i], lw=0.8, alpha=0.9, zorder=2)
t = X.groupby('xb', observed=False).ext.mean(); c.plot(xc, t.values, '-', color=INK, lw=2.0, label='All crests', zorder=4)
xs = np.logspace(-2.5, 2.5, 200); th = np.where(xs < 0.5, xs, 1 - 1 / (4 * xs)); c.plot(xs, th, '--', color=GREY, lw=1.0, label='Triangular crest', zorder=3)
c.axvline(1, color=GREY, lw=0.5, ls=':'); c.set_xscale('log'); c.set_xlim(10 ** -2.3, 10 ** 2.3); c.set_ylim(0, 1.03); c.xaxis.set_minor_locator(NullLocator())
c.set_xticks([0.01, 0.1, 1, 10, 100]); c.set_xticklabels(['0.01', '0.1', '1', '10', '100']); c.set_yticks([0, 0.25, 0.5, 0.75, 1])
c.set_xlabel('Crest duration / revisit interval'); c.set_ylabel('Crest extent captured')
c.legend(loc='upper left', fontsize=6.0, handlelength=1.6, bbox_to_anchor=(-0.02, 1.04))

# ---------------- d: schemes x area class, with crest-footprint residents
order = [('Landsat 8/9 (8 d)', 'Landsat 8/9, 8 d'), ('Sentinel-2 (5 d)', 'Sentinel-2, 5 d'), ('HLS (S2 + Landsat)', 'Landsat + Sentinel-2'), ('MODIS Terra+Aqua', 'MODIS Terra + Aqua'),
         ('Sentinel-1 (12 d)', 'Sentinel-1, 12 d'), ('Sentinel-1A/C (6 d)', 'Sentinel-1A/C, 6 d'), ('S1 6 d + NISAR', 'Sentinel-1 + NISAR'), ('All current', 'All current'),
         ('Daily SAR', 'Radar, daily'), ('MODIS, no cloud', 'MODIS, no cloud')]
P = S.dropna(subset=['pc'])
M = np.array([[S[(S.scheme == k) & (S.ab == b_)].ext.mean() for b_ in LABK] + [PCI.people[k]] for k, _ in order])
dd = ax_mm(fig, 37, T2, 63, H2)
im = dd.imshow(M, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]): dd.text(j, i, f'{M[i, j] * 100:.0f}', ha='center', va='center', fontsize=6.0, color='white' if M[i, j] > 0.55 else INK)
dd.axvline(6.5, color='white', lw=3)
for yv in (3.5, 6.5, 7.5): dd.axhline(yv, color='white', lw=2.2)
dd.set_xticks(range(8)); dd.set_xticklabels(['<30', '30–\n100', '100–\n300', '300–\n1k', '1k–\n3k', '3k–\n10k', '>10k', 'Resi-\ndents'], linespacing=0.9)
dd.set_yticks(range(len(order))); dd.set_yticklabels([n for _, n in order], fontsize=6.4); dd.tick_params(length=0)
for sp in dd.spines.values(): sp.set_visible(False)
dd.set_xlabel('Drainage area (km$^2$)', x=7 / 16)
fig.canvas.draw(); rd = fig.canvas.get_renderer()
xl = min(l.get_window_extent(rd).x0 for l in dd.get_yticklabels()); xl = dd.transAxes.inverted().transform((xl, 0))[0] - 0.015
for y0, y1, t_ in [(-0.4, 3.4, 'Optical'), (3.6, 6.4, 'Radar'), (6.6, 7.4, ''), (7.6, 9.4, 'Possible')]:
    dd.plot([xl, xl], [y0, y1], '-', color=INK2, lw=0.8, transform=dd.get_yaxis_transform(), clip_on=False)
    if t_: dd.text(xl - 0.015, (y0 + y1) / 2, t_, transform=dd.get_yaxis_transform(), rotation=90, ha='right', va='center', fontsize=6.2, color=INK2)
cb = cbar(fig, 104, T2 + 4, 1.8, H2 - 8, im, orient='vertical', ticks=[0, 0.5, 1], fs=6.0); cb.ax.set_yticklabels(['0', '50%', '100%'])
fig.text(105.5 / W, 1 - (T2 + 2.6) / H, 'Captured', ha='center', va='bottom', fontsize=6.0)

# ---------------- e: residents captured against revisit; missions at their nominal revisit
e = ax_mm(fig, 126, T2, 41, H2)
Wq = Wp.dropna(subset=['pc']); cur = Wq.groupby('rv').apply(lambda q: q['pop'].sum() / q.pc.sum())
e.plot(cur.index, cur.values, '-', color=INK, lw=1.8, label='No cloud', zorder=3)
nom = {'MODIS Terra+Aqua': 12, 'MODIS, no cloud': 12, 'Landsat 8/9 (8 d)': 192, 'Sentinel-2 (5 d)': 120, 'HLS (S2 + Landsat)': 24 / (1 / 5 + 1 / 8), 'Sentinel-1 (12 d)': 288,
       'Sentinel-1A/C (6 d)': 144, 'S1 6 d + NISAR': 96, 'All current': 24 / (2 + 1 / 5 + 1 / 8 + 1 / 6 + 1 / 12), 'Daily SAR': 24}
sty = {'MODIS Terra+Aqua': ('o', CLEAR), 'Landsat 8/9 (8 d)': ('o', CLEAR), 'Sentinel-2 (5 d)': ('o', CLEAR), 'HLS (S2 + Landsat)': ('o', CLEAR),
       'Sentinel-1 (12 d)': ('D', PRED), 'Sentinel-1A/C (6 d)': ('D', PRED), 'S1 6 d + NISAR': ('D', PRED), 'All current': ('s', OBS), 'Daily SAR': ('D', WATER_L), 'MODIS, no cloud': ('o', 'white')}
for k, x in nom.items():
    mk, col = sty[k]; y = PCI.people[k]
    e.errorbar(x, y, yerr=[[y - PCI.lo[k]], [PCI.hi[k] - y]], fmt=mk, ms=4.0, mfc=col, mec=INK, mew=0.5, color=INK2, lw=0.7, zorder=5)
xe = np.exp(np.interp(-PCI.people['MODIS Terra+Aqua'], -cur.values, np.log(cur.index.values)))
e.plot([12, xe], [PCI.people['MODIS Terra+Aqua']] * 2, ':', color='#A6611A', lw=1.0, zorder=2)
e.annotate('', xy=(14.5, PCI.people['MODIS Terra+Aqua'] + 0.03), xytext=(14.5, PCI.people['MODIS, no cloud'] - 0.04), arrowprops=dict(arrowstyle='->', lw=0.8, color=INK2))
e.text(16.5, 0.545, 'Cloud', fontsize=5.9, color=INK2, ha='left', va='center')
lbl = {'Landsat 8/9 (8 d)': ('L8/9', 1.25, 0.0, 'left'), 'Sentinel-2 (5 d)': ('S2', 1.0, -0.07, 'center'), 'HLS (S2 + Landsat)': ('L + S2', 0.85, 0.0, 'right'),
       'Sentinel-1 (12 d)': ('S1', 1.32, 0.0, 'left'), 'Sentinel-1A/C (6 d)': ('S1A/C', 1.32, 0.03, 'left'), 'S1 6 d + NISAR': ('S1 +\nNISAR', 1.14, 0.095, 'left'),
       'All current': ('All\ncurrent', 0.8, 0.0, 'right'), 'Daily SAR': ('Daily radar', 1.3, 0.02, 'left'), 'MODIS Terra+Aqua': ('MODIS', 0.8, 0.0, 'right'),
       'MODIS, no cloud': ('MODIS,\nno cloud', 0.8, 0.0, 'right')}
for k, (t_, fx, dy, ha) in lbl.items(): e.text(nom[k] * fx, PCI.people[k] + dy, t_, fontsize=5.8, ha=ha, va='center', color=INK, linespacing=0.95)
e.set_xscale('log'); e.set_xticks([1, 6, 24, 72, 288]); e.set_xticklabels(['1 h', '6 h', '1 d', '3 d', '12 d']); e.xaxis.set_minor_locator(NullLocator())
e.set_xlim(0.9, 450); e.set_ylim(0, 1.03); e.set_yticks([0, 0.25, 0.5, 0.75, 1]); e.set_xlabel('Revisit interval'); e.set_ylabel('Residents captured / at crest')
e.legend(loc='upper right', fontsize=6.0, handlelength=1.6, bbox_to_anchor=(1.03, 1.04))

for l_, x_, t_ in [('a', 1, T1 - 2), ('b', 58, T1 - 2), ('c', 115, T1 - 2), ('d', 1, T2 - 3), ('e', 115, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig11_osse')
print('slope', round(lr.slope, 3), 'r', round(lr.rvalue, 3), 'MODIS equiv d', round(xe / 24, 2))
