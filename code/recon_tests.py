# Reconstruction tests: baselines, timing from radar storm day, timing errors, leave-one-year-out rating curves, quantile intervals, error distributions, exclusions
import numpy as np, pandas as pd, time
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
t00=time.time()
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; BINS=[0,30,100,300,1000,3000,10000,1e7]
s=pd.read_parquet('data/main_sample.parquet'); P=pd.read_parquet('data/usgs/peaks.parquet'); P['site_no']=P.site_no.astype(str)
P=P[(~P.historic)&P.gage_ht_ft.notna()&(P.peak_va_cfs>0)&(P.peak_dt.astype(str).str[:4].astype(int)>=1990)]
pc=P.peak_cd.fillna('').astype(str); P=P[~pc.str.contains('5|6|3|8|4')]
s=s[s.q_at_first.notna()]; P=P[P.site_no.isin(s.site_no.unique())]
H0=None
def fit(h,qv):
    q=np.log(qv); best=None
    h0s=np.linspace(h.min()-40,h.min()-0.05,120)
    Y=np.log(h[None,:]-h0s[:,None]); A=np.vstack([np.ones_like(q),q]).T
    c=np.linalg.lstsq(A,Y.T,rcond=None)[0]   # (2,120)
    res=Y-(A@c).T; r2=1-(res**2).sum(1)/((Y-Y.mean(1,keepdims=True))**2).sum(1)
    okc=(c[1]>=0.15)&(c[1]<=0.8)
    if not okc.any(): return None
    k=np.argmax(np.where(okc,r2,-9))
    return (r2[k],h0s[k],c[0,k],c[1,k]) if r2[k]>=0.8 else None
fits={}; groups={site:g for site,g in P.groupby('site_no')}
for site,g in groups.items():
    if len(g)<10: continue
    f=fit(g.gage_ht_ft.values.astype(float),g.peak_va_cfs.values.astype(float))
    if f: fits[site]=f
print('rated',len(fits),'of',sum(len(g)>=10 for g in groups.values()),round(time.time()-t00),'s')
def H(f,Q): r2,h0,a,b=f; return h0+np.exp(a+b*np.log(Q))
# depth on every day offset 0..10 and crest depth, full rating and leave-own-peak-out rating
rows=[]; nlo=0
for r in s[s.site_no.isin(fits)].itertuples():
    out=[]
    for mode in ('full','loyo'):
        f=fits[r.site_no]
        if mode=='loyo':
            g=groups[r.site_no]; g=g[g.peak_dt.astype(str)!=r.peak_date]
            f=fit(g.gage_ht_ft.values.astype(float),g.peak_va_cfs.values.astype(float)) if len(g)>=9 else None
            if f is None: out+= [np.nan]*12; continue
        off=r.gage_ht_ft-H(f,r.peak_va_cfs) if np.isfinite(r.gage_ht_ft) else 0.0; hb=H(f,max(r.q_pre_min_cfs,1e-3))+off
        out.append(H(f,r.peak_va_cfs)+off-hb)
        for o in range(11):
            q=getattr(r,f'q{o}'); out.append(H(f,max(q*r.qmax_d,1e-3))+off-hb if np.isfinite(q) else np.nan)
    rows.append([r.site_no,r.peak_date,r.drain_area_km2,r.lat,r.lon,r.first_clear]+out)
cols=['site_no','peak_date','A','lat','lon','first_clear','dpk']+[f'd{o}' for o in range(11)]+['dpk_L']+[f'dL{o}' for o in range(11)]
D=pd.DataFrame(rows,columns=cols); D.to_parquet('data/rev_depth_days.parquet'); print('depth table',D.shape,round(time.time()-t00),'s')
