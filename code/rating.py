import numpy as np, pandas as pd
P=pd.read_parquet('data/usgs/peaks.parquet'); P['site_no']=P.site_no.astype(str)
P=P[(~P.historic)&P.gage_ht_ft.notna()&(P.peak_va_cfs>0)&(P.peak_dt.astype(str).str[:4].astype(int)>=1990)]
pc=P.peak_cd.fillna('').astype(str); P=P[~pc.str.contains('5|6|3|8|4')]
s=pd.read_parquet('data/main_sample.parquet'); s=s[s.q_at_first.notna()]
P=P[P.site_no.isin(s.site_no.unique())]
fits={}
for site,g in P.groupby('site_no'):
    if len(g)<10: continue
    h=g.gage_ht_ft.values.astype(float); q=np.log(g.peak_va_cfs.values.astype(float))
    best=None
    for h0 in np.linspace(h.min()-40,h.min()-0.05,120):
        y=np.log(h-h0); A=np.vstack([np.ones_like(q),q]).T; c,res,_,_=np.linalg.lstsq(A,y,rcond=None)
        r2=1-((y-A@c)**2).sum()/((y-y.mean())**2).sum()
        if 0.15<=c[1]<=0.8 and (best is None or r2>best[0]): best=(r2,h0,c[0],c[1])
    if best and best[0]>=0.8: fits[site]=best
print('sites with rating',len(fits),'of',s.site_no.nunique())
def H(site,Q):
    r2,h0,a,b=fits[site]; return h0+np.exp(a+b*np.log(Q))
rows=[]
for r in s[s.site_no.isin(fits)].itertuples():
    off=r.gage_ht_ft-H(r.site_no,r.peak_va_cfs) if np.isfinite(r.gage_ht_ft) else 0.0
    hp=H(r.site_no,r.peak_va_cfs)+off; q1=max(r.q_at_first*r.qmax_d,1e-3); qb=max(r.q_pre_min_cfs,1e-3)
    h1=H(r.site_no,q1)+off; hb=H(r.site_no,qb)+off
    rows.append((r.site_no,r.peak_date,r.drain_area_km2,r.abin,hp-hb,h1-hb,(h1-hb)/(hp-hb) if hp>hb else np.nan, hp-h1))
R=pd.DataFrame(rows,columns=['site_no','peak_date','A','abin','dh_peak_ft','dh_first_ft','depth_frac','drop_ft'])
R.to_parquet('data/stage_first.parquet')
g=R.groupby('abin',observed=True)
print(pd.DataFrame({'n':g.size(),'dh_peak_m':g.dh_peak_ft.median()*0.3048,'drop_m':g.drop_ft.median()*0.3048,'depth_frac_med':g.depth_frac.median(),'depth_frac_q25':g.depth_frac.quantile(.25),'depth_frac_q75':g.depth_frac.quantile(.75)}).round(2).to_string())
