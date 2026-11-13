import numpy as np, pandas as pd
P=P[(~P.historic)&P.gage_ht_ft.notna()&(P.peak_va_cfs>0)&(P.peak_dt.astype(str).str[:4].astype(int)>=1990)]
pc=P.peak_cd.fillna('').astype(str); P=P[~pc.str.contains('5|6|3|8|4')]
_s=pd.read_parquet('data/main_sample.parquet'); _s=_s[_s.q_at_first.notna()]
P=P[P.site_no.isin(_s.site_no.unique())]
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
print('sites with rating',len(fits),'of',_s.site_no.nunique())
def H(site,Q):
    r2,h0,a,b=fits[site]; return h0+np.exp(a+b*np.log(Q))
