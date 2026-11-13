# People inside the crest flood footprint that the first clear MODIS look still shows, worldwide.
# Chain: first clear look (max of Terra/Aqua clear fraction >= 0.4, calibrated to GOES) -> flow fraction q
#        -> depth fraction q**b (US rating exponent by area class) -> HAND x HRSL people below depth, above the HAND=0 network.
# Crest depth per area class from US annual-peak stage (median). The same chain is checked on US crests against stage-based exposure.
import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; BINS=[0,30,100,300,1000,3000,10000,1e7]
DH=np.array([0,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15]); cP=['P0']+[f'P{d}' for d in DH[1:]]
key=lambda o:'0' if o==0 else (f'm{-o}' if o<0 else f'p{o}')
def qfirst(d,th=0.4):
    qf=np.full(len(d),np.nan); done=np.zeros(len(d),bool)
    for o in range(0,11):
        b=np.fmax(d[f'T_{key(o)}'].values,d[f'A_{key(o)}'].values); q=d[f'q_{key(o)}'].values
        hit=(~done)&(b>=th)&np.isfinite(q); qf[hit]=q[hit]; done|=hit
    return qf
# US calibration of depth exponent and crest depth by area class
st=pd.read_parquet('data/stage_first.parquet'); s=pd.read_parquet('data/main_sample.parquet')
X=st.merge(s[['site_no','peak_date','q_at_first']],on=['site_no','peak_date'])
X=X[(X.q_at_first>0.02)&(X.q_at_first<0.98)&(X.depth_frac>0)]
X['b']=np.log(X.depth_frac.clip(1e-3,1))/np.log(X.q_at_first); X['ab']=pd.cut(X.A,BINS,labels=lab)
BEXP=X.groupby('ab',observed=True).b.median(); HC=st.assign(ab=pd.cut(st.A,BINS,labels=lab)).groupby('ab',observed=True).dh_peak_ft.median()*0.3048
print('depth exponent by area',BEXP.round(2).to_dict()); print('median crest depth (m)',HC.round(2).to_dict())
def expo(E,H,idcol):
    E=E.merge(H[[ 'site_no']+cP],left_on=idcol,right_on='site_no',how='inner')
    hc=HC.reindex(E.ab).values; df=np.clip(E.qf.values,0,1)**BEXP.reindex(E.ab).values
    V=E[cP].values.astype('f8')
    pc=np.array([np.interp(h,DH,v) for h,v in zip(np.clip(hc,0,15),V)])-V[:,0]; pf=np.array([np.interp(h,DH,v) for h,v in zip(np.clip(hc*df,0,15),V)])-V[:,0]
    E['P_crest']=pc; E['P_first']=pf; return E
# global
C=pd.read_parquet('data/global/caravan_ana.parquet'); C=C[C.rain].copy(); C['qf']=qfirst(C); C=C[np.isfinite(C.qf)]
HG=pd.read_parquet('data/global/hand_pop_caravan.parquet')
G=expo(C,HG,'basin_id')
# US check on MODIS chain: same people grid (US HAND x HRSL), chain vs stage-based answer
U=pd.read_parquet('data/global/usgs_goes_modis.parquet'); U['qf']=qfirst(U.rename(columns={f'q{o}':f'q_{key(o)}' for o in range(-3,11)}))
U['ab']=pd.cut(U.drain_area_km2,BINS,labels=lab); HU=pd.read_parquet('data/hand_pop.parquet'); UE=expo(U[np.isfinite(U.qf)],HU,'site_no')
EX=pd.read_parquet('data/exposure.parquet')
print('US stage-based seen share',round(EX.P_first.sum()/EX.P_crest.sum(),3),' US MODIS-chain seen share',round(UE.P_first.sum()/UE.P_crest.sum(),3))
rows=[]
for nm,d in [('US stage (GOES)',EX.assign(ab=EX.ab.astype(str)))]+[('US chain (MODIS)',UE)]+[(k,G[G.dataset==k]) for k in ['namerica','camelsgb','lamah','camelsbr','camelscl','camelsaus']]+[('All Caravan',G)]:
    for b in lab+['all']:
        dd=d if b=='all' else d[d.ab.astype(str)==b]
        if len(dd)<30 or dd.P_crest.sum()<=0: continue
        rows.append(dict(set=nm,ab=b,n=len(dd),people_crest=dd.P_crest.sum(),seen=dd.P_first.sum()/dd.P_crest.sum()))
R=pd.DataFrame(rows); R.to_csv('fig/exposure_global.csv',index=False)
print(R[R.ab=='all'].round(3).to_string()); print(R[R.set=='All Caravan'].round(3).to_string())
G[['basin_id','dataset','lat','lon','area_km2','ab','peak_date','qf','P_crest','P_first']].to_parquet('data/global/exposure_caravan.parquet')
