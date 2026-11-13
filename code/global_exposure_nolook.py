# Worldwide residents with crests that had no clear look within ten days counted as seen = 0
import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; BINS=[0,30,100,300,1000,3000,10000,1e7]
DH=np.array([0,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15]); cP=['P0']+[f'P{d}' for d in DH[1:]]
st=pd.read_parquet('data/stage_first.parquet'); HC=st.assign(ab=pd.cut(st.A,BINS,labels=lab)).groupby('ab',observed=True).dh_peak_ft.median()*0.3048
G=pd.read_parquet('data/global/exposure_caravan.parquet'); G=G[G.dataset!='camelsaus']
C=pd.read_parquet('data/global/caravan_ana.parquet'); C=C[C.rain&(C.dataset!='camelsaus')]
N=C.merge(G[['basin_id','peak_date']],on=['basin_id','peak_date'],how='left',indicator=True); N=N[N._merge=='left_only'].drop(columns='_merge')
HG=pd.read_parquet('data/global/hand_pop_caravan.parquet'); N=N.merge(HG[['site_no']+cP],left_on='basin_id',right_on='site_no',how='inner')
N['ab']=pd.cut(N.area_km2,BINS,labels=lab); hc=HC.reindex(N.ab.astype(str)).values; V=N[cP].values.astype('f8')
N['P_crest']=np.array([np.interp(h,DH,v) for h,v in zip(np.clip(hc,0,15),V)])-V[:,0]; N['P_first']=0.0
print('crests without a clear look',len(N),'person-events (M)',round(N.P_crest.sum()/1e6,1))
A=pd.concat([G[['ab','P_crest','P_first']].assign(ab=G.ab.astype(str)),N[['ab','P_crest','P_first']].assign(ab=N.ab.astype(str))])
print('conditional: P',round(G.P_crest.sum()/1e6,1),'seen',round(G.P_first.sum()/G.P_crest.sum(),3))
print('all storm-driven crests: P',round(A.P_crest.sum()/1e6,1),'seen',round(A.P_first.sum()/A.P_crest.sum(),3))
print('by class conditional',{b:round(G[G.ab.astype(str)==b].P_first.sum()/G[G.ab.astype(str)==b].P_crest.sum(),2) for b in lab})
print('by class all',{b:round(A[A.ab==b].P_first.sum()/A[A.ab==b].P_crest.sum(),2) for b in lab})
print('no-look share of crests',round(len(N)/(len(N)+len(G)),3),'of person-events',round(N.P_crest.sum()/A.P_crest.sum(),3))
