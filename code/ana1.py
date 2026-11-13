import numpy as np, pandas as pd
e=pd.read_parquet('data/matched.parquet')
OFF=range(-3,11)
for o in OFF: e[f'b{o}']=np.fmax(e[f't{o}'],e[f'a{o}'])
s=e[e.ge_median & ~e.flag_regulated & e.drain_area_km2.notna() & e.has_dv & e.qmax_d.notna()].copy()
print('main sample',len(s),'sites',s.site_no.nunique())
bins=[0,30,100,300,1000,3000,10000,100000,1e7]; lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','10k-100k','>100k']
s['abin']=pd.cut(s.drain_area_km2,bins,labels=lab)
# first clear look (best-of >=0.5) on day>=0
B=s[[f'b{o}' for o in range(0,11)]].values; Qr=s[[f'q{o}' for o in range(0,11)]].values
first=np.full(len(s),np.nan); qfirst=np.full(len(s),np.nan)
for i in range(len(s)):
    j=np.where(B[i]>=0.5)[0]
    if len(j): first[i]=j[0]; qfirst[i]=Qr[i,j[0]]
s['first_clear']=first; s['q_at_first']=qfirst
# near-crest capture (Q>=0.8 qmax) over days -3..10, expected vs climatology
Ball=s[[f'b{o}' for o in OFF]].values; Qall=s[[f'q{o}' for o in OFF]].values
near=(Qall>=0.8)
cap=1-np.nanprod(np.where(near,1-np.nan_to_num(Ball,nan=0),1),axis=1)
capc=1-np.prod(np.where(near,1-s.clim_b.values[:,None],1),axis=1)
s['cap']=cap; s['cap_clim']=capc; s['n_near']=near.sum(1)
g=s.groupby('abin',observed=True)
out=pd.DataFrame({'n':g.size(),'clear_t0_terra':g.t0.mean(),'clim_terra':g.clim_t.mean(),'clear_t0_best':g.b0.mean(),'clim_best':g.clim_b.mean(),
 'clear_t1_best':g.b1.mean(),'clear_t3_best':g.b3.mean(),'lag_first_med':g.first_clear.median(),'q_at_first_med':g.q_at_first.median(),'never10d':g.first_clear.apply(lambda x:x.isna().mean()),
 'n_near_med':g.n_near.median(),'cap':g.cap.mean(),'cap_clim':g.cap_clim.mean()})
out['ratio_t0']=out.clear_t0_best/out.clim_best; out['cap_ratio']=out.cap/out.cap_clim
pd.set_option('display.width',250); print(out.round(2))
print('all: t0 best',s.b0.mean().round(3),'clim',s.clim_b.mean().round(3))
# curve by offset for small vs large
for nm,sel in [('<100',s.drain_area_km2<100),('100-1k',s.drain_area_km2.between(100,1000)),('1k-10k',s.drain_area_km2.between(1000,1e4)),('>10k',s.drain_area_km2>1e4)]:
    print(nm, [round(np.nanmean(s.loc[sel,f'b{o}'])/np.nanmean(s.loc[sel,'clim_b']),2) for o in OFF])
s.to_parquet('data/main_sample.parquet')
