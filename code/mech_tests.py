# Stricter cross-fits of the two-timescale prediction (US): spatial blocks, years and storms as folds; area-blind, area-only and lag baselines; lag effect within area classes
import numpy as np, pandas as pd
M=pd.read_parquet('data/mech.parquet'); M['L']=M.lag+1
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
M['ab']=pd.cut(M.drain_area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=lab).astype(str)
M['season']=pd.to_datetime(M.peak_date).dt.month.map(lambda m:['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'][m-1])
M['reg']=np.where(M.lon<-100,'W','E'); M=M.reset_index(drop=True)
JP=np.arange(-2,8); B=np.stack([M[f'b{o}'].values for o in range(-3,11)],1)
o=M.kr.values[:,None]+JP[None,:]-1; ok=(o>=-3)&(o<=10); E=np.where(ok,np.take_along_axis(B,np.clip(o+3,0,13),1),np.nan); C=M.clim_b.values
def curve(idx):
    e=E[idx]; c=np.broadcast_to(C[idx,None],e.shape); m=np.isfinite(e); return np.where(m,e,0).sum(0)/np.where(m,c,0).sum(0)
jj=M.L.clip(0,4).values+2
# fold definitions
rng=np.random.default_rng(3)
def kfold(keys,k=5,seed=0):
    u=pd.unique(keys); r=np.random.default_rng(seed).permutation(len(u)); f=dict(zip(u,r%k)); return pd.Series(keys).map(f).values
blk=(np.floor(M.lat/2.5).astype(int).astype(str)+'_'+np.floor(M.lon/2.5).astype(int).astype(str)).values
rainday=(pd.to_datetime(M.peak_date)+pd.to_timedelta(M.kr,unit='D')).dt.strftime('%Y%m%d').values
storm=np.array([a+'_'+b for a,b in zip(rainday,(np.floor(M.lat/5).astype(int).astype(str)+'_'+np.floor(M.lon/5).astype(int).astype(str)).values)])
yr=M.peak_date.str[:4].values
schemes={'site halves':kfold(M.site_no.values,2,3),'site 5-fold':kfold(M.site_no.values,5,1),'2.5° blocks':kfold(blk,5,2),'storms':kfold(storm,5,4),'years':pd.Series(yr).map({y:i for i,y in enumerate(sorted(set(yr)))}).values}
print('n storms',len(set(storm)),'n blocks',len(set(blk)))
def run(fo):
    pl=np.full(len(M),np.nan); pnull=pl.copy(); parea=pl.copy()
    for (rg,se),d in M.groupby(['reg','season']):
        ii=d.index.values
        for f in np.unique(fo[ii]):
            tr=ii[fo[ii]!=f]; te=ii[fo[ii]==f]
            if len(tr)<30: continue
            pl[te]=curve(tr)[jj[te]]
            pnull[te]=M.b0.values[tr].sum()/C[tr][np.isfinite(M.b0.values[tr])].sum()
            for b in set(M.ab.values[te]):
                trb=tr[M.ab.values[tr]==b]; teb=te[M.ab.values[te]==b]
                if len(trb)>=10: parea[teb]=np.nansum(M.b0.values[trb])/C[trb][np.isfinite(M.b0.values[trb])].sum()
    X=M.assign(pl=pl,pn=pnull,pa=parea); G=[]
    for (b,se,rg),d in X.groupby(['ab','season','reg']):
        d=d[np.isfinite(d.pl)&np.isfinite(d.b0)]
        if len(d)<30: continue
        w=d.clim_b; G.append(dict(b=b,season=se,reg=rg,n=len(d),obs=d.b0.sum()/w.sum(),lag=(d.pl*w).sum()/w.sum(),null=(d.pn*w).sum()/w.sum(),area=(d.pa*w).sum()/w.sum() if d.pa.notna().all() else np.nan))
    G=pd.DataFrame(G); res={'groups':len(G)}
    for k in ['lag','null','area']:
        g=G.dropna(subset=[k]); res[f'r_{k}']=np.corrcoef(g.obs,g[k])[0,1]; res[f'rmse_{k}']=np.sqrt(((g.obs-g[k])**2).mean())
    return res,G
rows=[]
for nm,fo in schemes.items():
    res,G=run(fo); res['scheme']=nm; rows.append(res)
    if nm=='storms': G.to_csv('fig/rev/mech_cells_storms.csv',index=False)
Rt=pd.DataFrame(rows); print(Rt.round(3).to_string()); Rt.to_csv('fig/rev/mech_crossfit_US.csv',index=False)
# lag effect within area classes: observed crest-day ratio by area class x crest lag (days after storm day)
M['Lc']=M.L.clip(0,3)
W=M.groupby(['ab','Lc']).apply(lambda d:pd.Series({'n':len(d),'r0':d.b0.sum()/d.clim_b[d.b0.notna()].sum()})).reset_index()
print(W.pivot(index='ab',columns='Lc',values='r0').reindex(lab).round(2)); print(W.pivot(index='ab',columns='Lc',values='n').reindex(lab))
W.to_csv('fig/rev/mech_area_by_lag_US.csv',index=False)
# crest-level logistic-free check: within area class, Spearman of lag vs b0/clim
from scipy.stats import spearmanr
for b in lab:
    d=M[(M.ab==b)&M.b0.notna()]; print(b,len(d),'rho(lag, b0-clim)',round(spearmanr(d.L,d.b0-d.clim_b)[0],3))
