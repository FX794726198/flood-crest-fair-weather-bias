# Multi-look Bayesian recession: prior on log-decay rate k from (area, season, location); each clear look gives depth with noise;
# posterior crest depth by grid over (log d_c, k). Compare with first-look naive and prior-only.
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
s=pd.read_parquet('data/main_sample.parquet'); P=pd.read_parquet('data/usgs/peaks.parquet'); P['site_no']=P.site_no.astype(str)
exec(open('code/rating_fit.py').read())   # provides fits, H()
s=s[s.site_no.isin(fits)&s.first_clear.notna()].copy()
dep=np.full((len(s),11),np.nan)
for j,r in enumerate(s.itertuples()):
    off=r.gage_ht_ft-H(r.site_no,r.peak_va_cfs); hb=H(r.site_no,max(r.q_pre_min_cfs,1e-3))+off
    for o in range(11):
        q=getattr(r,f'q{o}')
        if np.isfinite(q): dep[j,o]=H(r.site_no,max(q*r.qmax_d,1e-3))+off-hb
s['dpk']=s.gage_ht_ft-(s.apply(lambda r:H(r.site_no,max(r.q_pre_min_cfs,1e-3))+r.gage_ht_ft-H(r.site_no,r.peak_va_cfs),axis=1))
clear=(s[[f'b{o}' for o in range(11)]].values>=0.5)
ok=(s.dpk>0.3).values; s=s[ok]; dep=dep[ok]; clear=clear[ok]
# training target for prior: event decay rate k from crest depth to depth on day 3 (or last valid), k=log(dpk/d3)/3
d3=np.where(np.isfinite(dep[:,3])&(dep[:,3]>0.02),dep[:,3],np.nan); kk=np.log(s.dpk.values/d3)/3
s['lA']=np.log10(s.drain_area_km2); s['mon']=s.peak_date.str[5:7].astype(int)
X=s[['lA','mon','lat','lon','inst_to_daily']].assign(inst_to_daily=np.nan).values  # inst_to_daily is unknown without gauge -> drop
X=s[['lA','mon','lat','lon']].values
pri_mu=np.full(len(s),np.nan); pri_sd=np.full(len(s),np.nan)
for tr,te in GroupKFold(5).split(X,kk,s.site_no):
    m=np.isfinite(kk[tr]); g=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0).fit(X[tr][m],kk[tr][m])
    res=kk[tr][m]-g.predict(X[tr][m]); pri_mu[te]=g.predict(X[te]); pri_sd[te]=np.std(res)
SIG=0.3/0.3048  # observation error of depth from imagery+HAND, 0.3 m in ft
lg=np.linspace(np.log(0.1),np.log(60),300); ks=np.linspace(-0.2,3.0,161)
LG,KS=np.meshgrid(lg,ks,indexing='ij')
est_first=np.full(len(s),np.nan); est_post=np.full(len(s),np.nan); nlook=np.zeros(len(s),int)
rng=np.random.default_rng(0)
for j in range(len(s)):
    looks=[o for o in range(11) if clear[j,o] and np.isfinite(dep[j,o])][:3]
    if not looks: continue
    nlook[j]=len(looks); obs=[(o,max(dep[j,o]+rng.normal(0,SIG),0.01)) for o in looks]
    est_first[j]=obs[0][1]
    lp=-0.5*((KS-pri_mu[j])/pri_sd[j])**2
    for o,y in obs: lp+=-0.5*((np.exp(LG-KS*o)-y)/SIG)**2
    w=np.exp(lp-lp.max()); est_post[j]=np.exp((w*LG).sum()/w.sum())
s['est_first']=est_first; s['est_post']=est_post; s['nlook']=nlook
s['ab']=pd.cut(s.drain_area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k'])
g=s[np.isfinite(s.est_post)].groupby('ab',observed=True)
o=pd.DataFrame({'n':g.size(),'looks_med':g.nlook.median(),'first_ratio':g.apply(lambda d:(d.est_first/d.dpk).median()),'post_ratio':g.apply(lambda d:(d.est_post/d.dpk).median()),
 'first_w25':g.apply(lambda d:(abs(d.est_first/d.dpk-1)<=.25).mean()),'post_w25':g.apply(lambda d:(abs(d.est_post/d.dpk-1)<=.25).mean()),
 'first_mae_m':g.apply(lambda d:abs(d.est_first-d.dpk).median()*.3048),'post_mae_m':g.apply(lambda d:abs(d.est_post-d.dpk).median()*.3048)})
print(o.round(2).to_string()); s[['site_no','peak_date','drain_area_km2','dpk','est_first','est_post','nlook']].to_parquet('data/recon2.parquet')
