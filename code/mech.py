import numpy as np, pandas as pd
s=pd.read_parquet('data/main_sample.parquet'); F=pd.read_parquet('data/mrms_5day.parquet')
M=s.merge(F[['site_no','peak_date']+[f'P{k}' for k in range(-3,2)]],on=['site_no','peak_date'])
P=M[[f'P{k}' for k in range(-3,2)]].values
ok=np.isfinite(P).all(1)&(np.nanmax(P,1)>=10); M=M[ok].copy(); P=P[ok]
M['kr']=np.argmax(P,1)-3          # rain-file date relative to crest date
M['lag']=-M.kr                     # crest date minus rain-file date (days)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; M['ab']=pd.cut(M.drain_area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
print('events',len(M)); print(pd.crosstab(M.ab,M.lag,normalize='index').round(2))
# rain-aligned composite clear-sky ratio by area bin, j = days after rain-file date
R={}
for b in lab:
    d=M[M.ab==b]; row=[]
    for j in range(0,6):
        vals=[];cl=[]
        for o_k in range(-3,2):
            dd=d[d.kr==o_k]; o=o_k+j
            if -3<=o<=10 and len(dd): vals.append(dd[f'b{o}'].values); cl.append(dd.clim_b.values)
        v=np.concatenate(vals); c=np.concatenate(cl); row.append(np.nanmean(v)/np.nanmean(c))
    R[b]=row
C=pd.DataFrame(R,index=[f'j{j}' for j in range(6)]).T; print('rain-aligned clear/normal'); print(C.round(2))
# pooled curve and prediction of crest-day clear ratio
pool=[]
for j in range(0,6):
    vals=[];cl=[]
    for o_k in range(-3,2):
        dd=M[M.kr==o_k]; o=o_k+j
        if -3<=o<=10 and len(dd): vals.append(dd[f'b{o}'].values); cl.append(dd.clim_b.values)
    pool.append(np.nanmean(np.concatenate(vals))/np.nanmean(np.concatenate(cl)))
pool=np.array(pool); print('pooled',pool.round(2))
M['pred']=pool[np.clip(M.lag.values,0,5)]
g=M.groupby('ab',observed=True)
print(pd.DataFrame({'n':g.size(),'obs':g.b0.mean()/g.clim_b.mean(),'pred':g.pred.mean(),'lag_mean':g.lag.mean(),'lag0_share':g.lag.apply(lambda x:(x<=0).mean())}).round(2))
M.to_parquet('data/mech.parquet'); pd.Series(pool).to_csv('data/pool_curve.csv')
