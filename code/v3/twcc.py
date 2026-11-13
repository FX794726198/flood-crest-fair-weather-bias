# Timing-weighted crest correction (TWCC)
# Two estimates of crest depth for every crest:
#   image branch  ln d_I = ln d_look + f(tau, A, month, lat, lon)   (learned recession over the wait tau since the crest)
#   prior branch  ln d_P = g(A, month, lat, lon)                    (no image)
# fused as ln d = w ln d_I + (1 - w) ln d_P with the weight that minimises the mean squared log error of the combination,
#   w = (s_P^2 - c) / (s_I^2 + s_P^2 - 2c),
# where s_I^2, s_P^2 and c are the mean squared log errors and the mean error product (uncentred moments) of the two branches in held-out gauges.
# The weight is estimated in cells of area class x wait and summarised by a logistic law in log10 A and ln(tau + 1),
#   equivalent to w = 1 / [1 + ((tau + 1) / T_w)^k] with T_w = alpha A^beta (Eq. 3 of the paper).
# Everything is nested: outer folds grouped by gauge evaluate, inner folds on the training gauges give the residuals
# that set the weights and the cross-validated residual-quantile intervals (empirical quantiles of inner held-out errors, by area class).
import sys, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from scipy.optimize import least_squares
FT = 0.3048
LAB = ['<30', '30-100', '100-300', '300-1k', '1k-3k', '3k-10k', '>10k']; BINS = [0, 30, 100, 300, 1000, 3000, 10000, 1e7]
IMG = ['lag', 'lA', 'mon', 'lat', 'lon']; PRI = ['lA', 'mon', 'lat', 'lon']
def fit(X, y): return HGB(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
def tbin(t): return np.clip(np.floor(t), 0, 4).astype(int)
def cells(d): return pd.Index([f'{a}|{b}' for a, b in zip(d.ab.values, tbin(d.lag.values))])

def moments(T, ri, rp):
    k = cells(T); df = pd.DataFrame({'vi': ri ** 2, 'vp': rp ** 2, 'c': ri * rp, 'n': 1}, index=k)
    g = df.groupby(level=0).agg({'vi': 'mean', 'vp': 'mean', 'c': 'mean', 'n': 'sum'})
    return g, (np.mean(ri ** 2), np.mean(rp ** 2), np.mean(ri * rp))

def weight_of(g, glob, E):
    k = cells(E); vi = g.vi.reindex(k).fillna(glob[0]).values; vp = g.vp.reindex(k).fillna(glob[1]).values; c = g.c.reindex(k).fillna(glob[2]).values
    return np.clip((vp - c) / np.maximum(vi + vp - 2 * c, 1e-6), 0, 1)

def law_fit(g):
    """logistic law w = 1/(1+exp(-(a0 + a1 log10 A + a2 ln(tau+0.5)))) fitted to cell weights, weighted by cell size"""
    a_ = np.array([np.log10([15, 55, 170, 550, 1700, 5500, 30000])[LAB.index(s.split('|')[0])] for s in g.index])
    t_ = np.array([int(s.split('|')[1]) for s in g.index]) + 0.5
    w_ = np.clip((g.vp - g.c) / np.maximum(g.vi + g.vp - 2 * g.c, 1e-6), 0, 1).values; n_ = g.n.values
    def res(p): return np.sqrt(n_) * (1 / (1 + np.exp(-(p[0] + p[1] * a_ + p[2] * np.log(t_ + 0.5)))) - w_)
    return least_squares(res, [0, 1, -1]).x

def law(p, lA, lag): return 1 / (1 + np.exp(-(p[0] + p[1] * lA + p[2] * np.log(np.clip(lag, 0, None) + 1.0))))

def twcc(M, look='d1', lagcol='lag', shift=0.0):
    """nested evaluation; returns a frame with branch, fused (cell and law weights) estimates and intervals"""
    M = M.copy(); M['lag'] = M[lagcol]; M['ldpk'] = np.log(M.dpk); M['ll'] = np.log(M[look]); M['y'] = M.ldpk - M.ll
    out = {k: np.full(len(M), np.nan) for k in ['li', 'lp', 'lw', 'll_law', 'w', 'wl', 'lo80', 'hi80', 'lo90', 'hi90']}
    laws = []
    for tr, te in GroupKFold(5).split(M, M.y, M.site_no):
        T = M.iloc[tr].reset_index(drop=True); E = M.iloc[te].copy(); E['lag'] = np.clip(E.lag + shift, 0, None)
        pi = np.zeros(len(T)); pp = pi.copy()
        for a_, b_ in GroupKFold(5).split(T, T.y, T.site_no):
            pi[b_] = T.ll.values[b_] + fit(T[IMG].values[a_], T.y.values[a_]).predict(T[IMG].values[b_])
            pp[b_] = fit(T[PRI].values[a_], T.ldpk.values[a_]).predict(T[PRI].values[b_])
        ri = T.ldpk.values - pi; rp = T.ldpk.values - pp
        g, glob = moments(T, ri, rp); p = law_fit(g); laws.append(p)
        li = E.ll.values + fit(T[IMG].values, T.y.values).predict(E[IMG].values); lp = fit(T[PRI].values, T.ldpk.values).predict(E[PRI].values)
        w = weight_of(g, glob, E); wl = law(p, E.lA.values, E.lag.values)
        # residual-quantile intervals for the fused estimate (law weights) from inner held-out residuals, by area class
        wt = law(p, T.lA.values, T.lag.values); rf = T.ldpk.values - (wt * pi + (1 - wt) * pp)
        for b in set(E.ab.values):
            q = np.quantile(rf[T.ab.values == b], [0.1, 0.9, 0.05, 0.95]); k = E.ab.values == b; f_ = (wl * li + (1 - wl) * lp)[k]
            idx = np.asarray(te)[k]
            out['lo80'][idx] = f_ + q[0]; out['hi80'][idx] = f_ + q[1]; out['lo90'][idx] = f_ + q[2]; out['hi90'][idx] = f_ + q[3]
        out['li'][te] = li; out['lp'][te] = lp; out['lw'][te] = w * li + (1 - w) * lp; out['ll_law'][te] = wl * li + (1 - wl) * lp
        out['w'][te] = w; out['wl'][te] = wl
    R = M[['site_no', 'peak_date', 'A', 'ab', 'lag', 'dpk', look]].copy()
    R['e_look'] = M[look]; R['e_img'] = np.exp(out['li']); R['e_pri'] = np.exp(out['lp']); R['e_tw'] = np.exp(out['lw']); R['e_law'] = np.exp(out['ll_law'])
    R['w'] = out['w']; R['w_law'] = out['wl']
    for k in ['lo80', 'hi80', 'lo90', 'hi90']: R[k] = np.exp(out[k])
    return R, np.array(laws)

def summ(d, est):
    r = d[est] / d.dpk
    return pd.Series({'n': len(d), 'w25': (np.abs(r - 1) <= 0.25).mean(), 'mae_m': np.median(np.abs(d[est] - d.dpk)) * FT, 'med_ratio': r.median(),
                      'over': (r > 1.25).mean(), 'under': (r < 0.75).mean()})

if __name__ == '__main__':
    M = pd.read_parquet('data/rev_recon.parquet').reset_index(drop=True); M['ab'] = pd.cut(M.A, BINS, labels=LAB).astype(str)
    R, laws = twcc(M)
    print('law coefficients by fold (a0, a1 per log10 A, a2 per ln(tau+1)):\n', laws.round(2))
    E = ['e_look', 'e_img', 'e_pri', 'e_tw', 'e_law']
    print(pd.DataFrame({e: summ(R, e) for e in E}).T.round(3).to_string())
    print(pd.DataFrame({e: [summ(R[R.ab == b], e).w25 for b in LAB] for e in E}, index=LAB).round(2).to_string())
    print(pd.DataFrame({e: [summ(R[R.ab == b], e).mae_m for b in LAB] for e in E}, index=LAB).round(2).to_string())
    R['c80'] = (R.dpk >= R.lo80) & (R.dpk <= R.hi80); R['c90'] = (R.dpk >= R.lo90) & (R.dpk <= R.hi90); R['wd80'] = R.hi80 / R.lo80
    print('coverage', R.c80.mean().round(3), R.c90.mean().round(3)); print(R.groupby('ab').agg(c80=('c80', 'mean'), c90=('c90', 'mean'), wd80=('wd80', 'median')).reindex(LAB).round(2).T.to_string())
    R['tb'] = tbin(R.lag.values); print(R.pivot_table(index='ab', columns='tb', values='w', aggfunc='median').reindex(LAB).round(2).to_string())
    R.to_parquet('data/v3_twcc.parquet'); np.save('data/v3_twcc_laws.npy', laws)
    # timing errors of one day
    for sh in (1, -1):
        Rs, _ = twcc(M, shift=sh); print('wait shifted', sh, summ(Rs, 'e_law').round(3).to_dict())
    # radar storm-day timing, no gauge timing
    S = pd.read_parquet('data/rev_recon_storm.parquet').reset_index(drop=True); S['ab'] = pd.cut(S.A, BINS, labels=LAB).astype(str)
    Rs, lws = twcc(S, look='dlook', lagcol='lag_storm')
    print('storm-timed', pd.DataFrame({e: summ(Rs, e) for e in E}).T.round(3).to_string())
    print(pd.DataFrame({e: [summ(Rs[Rs.ab == b], e).w25 for b in LAB] for e in E}, index=LAB).round(2).to_string())
    Rs.to_parquet('data/v3_twcc_storm.parquet')
    # rating curves refitted without the evaluated crest
    L = M[np.isfinite(M.dpk_L) & np.isfinite(M.d1L) & (M.dpk_L > 0.3) & (M.d1L > 0.05)].reset_index(drop=True).copy()
    L['dpk'] = L.dpk_L; L['d1'] = L.d1L
    RL, _ = twcc(L); print('leave-own-crest-out rating', summ(RL, 'e_law').round(3).to_dict(), 'first', summ(RL, 'e_look').round(3).to_dict())
    # crests with no clear look within ten days: the law gives w -> 0 and the correction is the prior
    D = pd.read_parquet('data/rev_depth_days.parquet'); D['lA'] = np.log10(D.A); D['mon'] = D.peak_date.str[5:7].astype(int)
    N = D[D.first_clear.isna() & (D.dpk > 0.3)].copy(); N['ab'] = pd.cut(N.A, BINS, labels=LAB).astype(str)
    C = pd.concat([M[['site_no', 'dpk'] + PRI].assign(nolook=False), N[['site_no', 'dpk'] + PRI].assign(nolook=True)], ignore_index=True)
    pr = np.zeros(len(C))
    for tr, te in GroupKFold(5).split(C, C.dpk, C.site_no):
        T = C.iloc[tr]; T = T[~T.nolook]; pr[te] = fit(T[PRI].values, np.log(T.dpk.values)).predict(C[PRI].values[te])
    N['e_pri'] = np.exp(pr[C.nolook.values]); print('no clear look within 10 days:', len(N), 'prior within 25%', round(((N.e_pri / N.dpk - 1).abs() <= .25).mean(), 3), 'mae', round(np.median(np.abs(N.e_pri - N.dpk)) * FT, 2))
    print(N.groupby('ab').size().reindex(LAB).to_dict())
    # residents in the HAND footprint at the corrected depth
    X = pd.read_parquet('data/exposure.parquet').merge(R[['site_no', 'peak_date', 'e_law', 'e_tw', 'dpk']], on=['site_no', 'peak_date'])
    DH = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15]); cP = ['P0'] + [f'P{d}' for d in DH[1:]]
    V = X[cP].values; hh = np.clip(X.e_law.values * FT, 0, 15); X['P_tw'] = np.array([np.interp(a, DH, v) for a, v in zip(hh, V)]) - X.P0
    print('depth check |dh_peak - dpk| max ft', np.abs(X.dh_peak_ft - X.dpk).max().round(4))
    X['ab'] = X.ab.astype(str).str.replace('–', '-')
    sh = lambda d, c: d[c].sum() / d.P_crest.sum()
    print('residents: first', round(sh(X, 'P_first'), 3), 'old rebuild', round(sh(X, 'P_gb'), 3), 'TWCC', round(sh(X, 'P_tw'), 3))
    print(pd.DataFrame({c: [sh(X[X.ab == b], c) for b in LAB] for c in ['P_first', 'P_gb', 'P_tw']}, index=LAB).round(2).to_string())
    x = X[X.P_crest > 10]
    for c in ['P_first', 'P_gb', 'P_tw']:
        r = x[c] / x.P_crest; print(c, 'single crests >10 residents within 25%', round(((r - 1).abs() <= .25).mean(), 3), 'over', round((r > 1.25).mean(), 3), 'under', round((r < .75).mean(), 3), 'n', len(x))
    print(pd.DataFrame({c: [(((x[x.ab == b][c] / x[x.ab == b].P_crest) - 1).abs() <= .25).mean() for b in LAB] for c in ['P_first', 'P_gb', 'P_tw']}, index=LAB).round(2).to_string())
    X.to_parquet('data/v3_exposure.parquet')
    for site, dt in [('02124269', '2020-11-12'), ('11466170', '2021-10-24'), ('05464500', '2019-03-19')]:
        r = X[(X.site_no == site) & (X.peak_date == dt)].iloc[0]
        print(site, dt, 'crest m', round(r.dpk * FT, 2), 'first', round(r.dh_first_ft * FT, 2), 'old', round(r.dpk_gb * FT, 2), 'TWCC', round(r.e_law * FT, 2), 'res first', round(r.P_first / r.P_crest, 2), 'old', round(r.P_gb / r.P_crest, 2), 'TWCC', round(r.P_tw / r.P_crest, 2))

