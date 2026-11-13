# Global test (Caravan x MODIS 1 deg x ERA5-Land): area dependence, storm-aligned cloud clearing, two-timescale prediction
import numpy as np, pandas as pd
rng=np.random.default_rng(5)
C=pd.read_parquet('data/global/caravan_matched.parquet')
C=C[C.weibull_p>=0.5].reset_index(drop=True)                      # crests at or above the median annual maximum
P=C[['P_m3','P_m2','P_m1','P_0','P_p1']].values; C['Pmax']=P.max(1); C['ks']=P.argmax(1)-3; C['L']=-C.ks
C['rain']=C.Pmax>=10
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; C['ab']=pd.cut(C.area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
C['month']=pd.to_datetime(C.peak_date).dt.month
south=C.lat<0; C['season']=((C.month%12)//3).map({0:'DJF',1:'MAM',2:'JJA',3:'SON'})
C.loc[south,'season']=C.loc[south,'season'].map({'DJF':'JJA','JJA':'DJF','MAM':'SON','SON':'MAM'})   # local season
OFF=list(range(-3,11)); key=lambda o:'0' if o==0 else (f'm{-o}' if o<0 else f'p{o}')
T=np.stack([C[f'T_{key(o)}'].values for o in OFF],1); A=np.stack([C[f'A_{key(o)}'].values for o in OFF],1)
cT=C.clim_T.values; cA=C.clim_A.values
JP=np.arange(-3,8)
o=C.ks.values[:,None]+JP[None,:]; ok=(o>=-3)&(o<=10); ii=np.clip(o+3,0,13)
ET=np.where(ok,np.take_along_axis(T,ii,1),np.nan); EA=np.where(ok,np.take_along_axis(A,ii,1),np.nan)
def curve(idx):
    out=[]
    for E,c in ((ET,cT),(EA,cA)):
        e=E[idx]; cc=np.broadcast_to(c[idx,None],e.shape); m=np.isfinite(e)&np.isfinite(cc); out.append(np.where(m,e,0).sum(0)/np.where(m,cc,0).sum(0))
    return 0.5*(out[0]+out[1])
def r0(d): return 0.5*(d.T_0.mean()/d.clim_T.mean()+d.A_0.mean()/d.clim_A.mean())
def tclear(cv):
    k0=np.where(JP==0)[0][0]
    for k in range(k0,len(JP)-1):
        if cv[k]<1<=cv[k+1]: return JP[k]+(1-cv[k])/(cv[k+1]-cv[k])
    return np.nan
ds_order=['camels','hysets','camelsgb','lamah','camelsbr','camelscl','camelsaus']
# 1) area dependence by dataset, rain-driven vs no-rain crests
A1=C[C.rain].groupby(['dataset','ab'],observed=True).apply(lambda d:pd.Series({'n':len(d),'r0':r0(d)})).reset_index()
A0=C[C.Pmax<5].groupby('ab',observed=True).apply(lambda d:pd.Series({'n':len(d),'r0':r0(d)})).reset_index()
A1r=C[C.rain].groupby('ab',observed=True).apply(lambda d:pd.Series({'n':len(d),'r0':r0(d)})).reset_index()
print(A1r.round(2).to_string()); print('no-rain',A0.round(2).to_string())
# 2) storm-aligned curves by dataset (rain-driven, storm day on or before crest)
R=C[C.rain&(C.ks<=0)].reset_index(drop=True); idxR=np.where(C.rain&(C.ks<=0))[0]
cur={}; TC={}
for ds in ds_order:
    ix=idxR[R.dataset.values==ds]; cur[ds]=curve(ix); TC[ds]=tclear(cur[ds])
cur['all']=curve(idxR); TC['all']=tclear(cur['all'])
print('storm-aligned clear/normal'); print(pd.DataFrame(cur,index=JP).T.round(2).to_string()); print('tclear',{k:round(v,2) for k,v in TC.items()})
lag=R.groupby(['dataset','ab'],observed=True).L.mean().unstack().reindex(ds_order); print('mean crest lag (d)'); print(lag.round(2).to_string())
# 3) cross-fitted prediction: dataset x season curve from the other half of basins
bas=C.basin_id.unique(); fold=dict(zip(bas,rng.integers(0,2,len(bas)))); fo=C.basin_id.map(fold).values
pred=np.full(len(C),np.nan); Lc=np.clip(C.L.values,0,3); jj=Lc+3
for (ds,se),d in C[C.rain&(C.ks<=0)].groupby(['dataset','season']):
    ix=d.index.values
    for f in (0,1):
        tr=ix[fo[ix]!=f]; te=ix[fo[ix]==f]
        if len(tr)<50: continue
        cv=curve(tr); pred[te]=cv[jj[te]]
C['pred']=pred
G=[]
for (ds,b,se),d in C[C.rain&(C.ks<=0)&np.isfinite(C.pred)].groupby(['dataset','ab','season'],observed=True):
    if len(d)<40: continue
    w=0.5*(d.clim_T+d.clim_A); G.append(dict(dataset=ds,b=b,season=se,n=len(d),obs=r0(d),pred=(d.pred*w).sum()/w.sum()))
G=pd.DataFrame(G); r=np.corrcoef(G.obs,G.pred)[0,1]; rmse=np.sqrt(((G.obs-G.pred)**2).mean())
print('groups',len(G),'r',round(r,3),'rmse',round(rmse,3))
# area-blind null: dataset x season mean crest-day ratio
G['null']=G.groupby(['dataset','season']).obs.transform('mean'); print('area-blind null r',round(np.corrcoef(G.obs,G.null)[0,1],3))
C.to_parquet('data/global/caravan_ana.parquet'); A1.to_csv('fig/fig4_area_by_dataset.csv',index=False); A0.to_csv('fig/fig4_norain.csv',index=False)
pd.DataFrame(cur,index=JP).to_csv('fig/fig4_curves.csv'); G.to_csv('fig/fig4_cells.csv',index=False); lag.to_csv('fig/fig4_lag.csv')
pd.Series(TC).to_csv('fig/fig4_tclear.csv')
