# Fig. 5  Fifteen-minute records of 392 crests: timing, persistence and storm alignment of cloud
from cstyle import *
M = pd.read_parquet('data/iv_capture_ext.parquet'); M['ab'] = M.ab.astype(str); M['lst'] = (M.t_crest_h + M.lon / 15) % 24
N = pd.read_parquet('data/iv_null.parquet')[['site_no', 'peak_date', 'ext_null']]; M = M.merge(N, on=['site_no', 'peak_date'], how='left')
H = 108; fig = figure(H)
T1, H1 = 4, 39; T2, H2 = 60, 38

# ---------------- a: captured crest extent under four cloud cases
a = ax_mm(fig, 13, T1, 68, H1)
cases = [('ext_cloudfree_max', 'No cloud', '#9CB3C9', 'o'), ('ext_clim_expected', 'Climatological cloud', '#5B7FA6', 's'),
         ('ext_null', 'Same-dates cloud', '#2F4B6E', 'D'), ('ext_observed_max', 'Observed cloud', OBS, 'o')]
med = {c: np.array([M[M.ab == k][c].median() for k in LABK]) for c, *_ in cases}
a.fill_between(XC, med['ext_observed_max'], med['ext_null'], where=med['ext_null'] > med['ext_observed_max'], color=OBS, alpha=0.12, lw=0, label='Lost to storm cloud')
for c, nm, col, mk in cases:
    a.plot(XC, med[c], '-', color=col, lw=1.3, label=nm, zorder=3); a.plot(XC, med[c], mk, color=col, ms=3.2 if mk != 'D' else 2.8, mec='white', mew=0.4, zorder=4)
area_axis(a); a.set_ylim(0, 1.04); a.set_yticks([0, 0.25, 0.5, 0.75, 1]); a.set_ylabel('Crest extent captured (median)')
hh, ll = a.get_legend_handles_labels(); a.legend(hh[1:] + hh[:1], ll[1:] + ll[:1], loc='lower right', fontsize=6.2, bbox_to_anchor=(1.03, -0.02))

# ---------------- b: wait from crest to first clear overpass
b = ax_mm(fig, 99, T1, 68, H1)
D = [M[M.ab == k].first_after_lag_h.dropna().values for k in LABK]
vp = b.violinplot([np.log10(np.clip(d, 0.3, None)) for d in D], positions=range(7), widths=0.78, showextrema=False)
for i, pc in enumerate(vp['bodies']): pc.set_facecolor(ACOL[i]); pc.set_alpha(0.55); pc.set_edgecolor('none')
md = np.array([np.median(d) for d in D])
for i, d in enumerate(D):
    q1, q3 = np.percentile(d, [25, 75]); b.plot([i, i], np.log10([q1, q3]), color=INK, lw=1.6, solid_capstyle='butt', zorder=3)
b.plot(range(7), np.log10(md), 'o', color='white', mec=INK, mew=0.8, ms=3.4, zorder=4)
for i, m in enumerate(md): b.text(i + 0.36, np.log10(m), f'{m:.0f}', ha='left', va='center', fontsize=6.0, color=INK)
b.set_yticks(np.log10([1, 3, 12, 24, 72, 168])); b.set_yticklabels(['1', '3', '12', '24', '72', '168']); b.set_ylim(np.log10(0.5), np.log10(400))
class_ticks(b); b.set_ylabel('Crest to first clear overpass (h)')

# ---------------- c: depth left at the first clear overpass against the wait
c = ax_mm(fig, 13, T2, 68, H2)
d = M.dropna(subset=['first_after_lag_h', 'first_after_df']).copy(); d['ci'] = d.ab.map({k: i for i, k in enumerate(LABK)})
c.scatter(d.first_after_lag_h, d.first_after_df.clip(0, 1), c=[ACOL[i] for i in d.ci], s=8, lw=0.25, edgecolors='white', alpha=0.9, zorder=3)
bins = np.array([0, 6, 12, 24, 48, 96, 300]); xm = np.sqrt(bins[1:] * np.maximum(bins[:-1], 1.5))
for sel, col, nm in [(d.A < 100, ACOL[0], '<100 km$^2$'), (d.A > 1000, ACOL[6], '>1,000 km$^2$')]:
    x = d[sel]; g = x.groupby(pd.cut(x.first_after_lag_h, bins), observed=False).first_after_df.median().clip(0, 1)
    c.plot(xm, g.values, '-', color='white', lw=3.4, zorder=4); c.plot(xm, g.values, '-', color=col, lw=1.8, label=f'Median, {nm}', zorder=5)
c.set_xscale('log'); c.set_xticks([1, 3, 12, 24, 72, 168]); c.set_xticklabels(['1', '3', '12', '24', '72', '168']); c.xaxis.set_minor_locator(NullLocator())
c.set_xlim(0.6, 300); c.set_ylim(-0.03, 1.05); c.set_yticks([0, 0.25, 0.5, 0.75, 1])
c.set_xlabel('Crest to first clear overpass (h)'); c.set_ylabel('Depth left / crest depth')
c.legend(loc='lower left', fontsize=6.2, bbox_to_anchor=(-0.01, -0.02))

# ---------------- d: crest time of day
e = ax_mm(fig, 99, T2, 68, H2)
hb = np.arange(0, 25, 2)
for sel, col, nm in [(M.A < 100, ACOL[0], '<100 km$^2$'), ((M.A >= 100) & (M.A <= 1000), ACOL[2], '100–1,000 km$^2$'), (M.A > 1000, ACOL[6], '>1,000 km$^2$')]:
    h, _ = np.histogram(M[sel].lst, bins=hb); e.step(hb, np.r_[h, h[-1]] / h.sum(), where='post', color=col, lw=1.4, label=f'{nm} ({sel.sum()})')
for t, nm in [(10.5, 'Terra'), (13.5, 'Aqua')]:
    e.axvline(t, color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1); e.text(t, 0.248, nm, ha='center', va='bottom', fontsize=FSS, color=INK2)
e.axhline(1 / 12, color=GREY, lw=0.6, ls=':'); e.set_xlim(0, 24); e.set_xticks([0, 6, 12, 18, 24]); e.set_ylim(0, 0.245)
e.set_xlabel('Crest time (local solar hour)'); e.set_ylabel('Share of crests')
e.legend(loc='upper left', fontsize=6.2, bbox_to_anchor=(-0.01, 1.0), ncol=1)

for l_, x_, t_ in [('a', 1, T1 - 2), ('b', 86, T1 - 2), ('c', 1, T2 - 3), ('d', 86, T2 - 3)]: letter(fig, l_, x_, t_)
save(fig, 'Fig05_fifteen_minute')
print(pd.DataFrame(med, index=LABK).round(2))
