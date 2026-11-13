# How much does a 1-degree cloud product dilute the crest-day clear-sky deficit? Coarsen GOES 0.1 deg to 1 deg and redo the crest-day ratio.
import numpy as np, pandas as pd
days=np.load('data/days.npy',allow_pickle=True); s=pd.read_parquet('data/main_sample.parquet')
out={}
for nm in ['terra','aqua']:
    X=np.load(f'data/{nm}.npy',mmap_mode='r'); nd=X.shape[0]; C=np.full((nd,24,59),np.nan,'f4')
    for i0 in range(0,nd,256):
        a=X[i0:i0+256,5:245,:].astype('f4'); a[a==255]=np.nan; a/=200
        a=a.reshape(a.shape[0],24,10,59,10); C[i0:i0+256]=np.nanmean(a,axis=(2,4))
    out[nm]=C
np.savez_compressed('data/goes_1deg.npz',terra=out['terra'],aqua=out['aqua'])
d=pd.to_datetime(pd.Series(days)); mon=d.dt.month.values
B=np.fmax(out['terra'],out['aqua'])
clim=np.stack([np.nanmean(B[mon==m],0) for m in range(1,13)])
didx={pd.Timestamp(x):i for i,x in enumerate(d)}
s=s[(s.lat>=25)&(s.lat<49)].copy(); s['I']=((s.lat-25)//1).astype(int); s['J']=((s.lon+125)//1).astype(int)
pk=pd.to_datetime(s.peak_date); s['m']=pk.dt.month
vals={}
for o in range(-3,11):
    ii=np.array([didx.get(t+pd.Timedelta(days=o),-1) for t in pk]); v=np.full(len(s),np.nan); ok=ii>=0
    v[ok]=B[ii[ok],s.I.values[ok],s.J.values[ok]]; s[f'c{o}']=v
s['clim1']=clim[s.m.values-1,s.I.values,s.J.values]
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; s['ab']=pd.cut(s.drain_area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
g=s.groupby('ab',observed=True)
R=pd.DataFrame({'n':g.size(),'goes01':g.b0.mean()/g.clim_b.mean(),'goes1deg':g.c0.mean()/g.clim1.mean()})
R['dilution']=(1-R.goes1deg)/(1-R.goes01); print(R.round(3).to_string())
s[['site_no','peak_date','I','J','clim1']+[f'c{o}' for o in range(-3,11)]].to_parquet('data/goes1deg_events.parquet'); R.to_csv('fig/goes_1deg_dilution.csv')
