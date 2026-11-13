# Sensitivity of the HAND footprint and residents to a vertical offset delta between the pre-event water surface and the HAND drainage
import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
r=pd.read_parquet('data/recon_cv.parquet'); H=pd.read_parquet('data/hand_pop.parquet')
DH=np.array([0,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15]); cA=['A0']+[f'A{d}' for d in DH[1:]]; cP=['P0']+[f'P{d}' for d in DH[1:]]
X=r.merge(H[['site_no']+cA+cP],on='site_no'); X['ab']=pd.cut(X.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab).astype(str)
VP=X[cP].values.astype(float)
def P(h):
    hh=np.clip(h,0,15); return np.array([np.interp(a,DH,v) for a,v in zip(hh,VP)])
rows=[]
for dl in [-1,-0.5,0,0.5,1]:
    base=P(np.full(len(X),max(dl,0)))
    pc=P(X.dh_peak_ft*0.3048+dl)-base; pf=P(X.dh_first_ft*0.3048+dl)-base; pg=P(X.dpk_gb*0.3048+dl)-base
    pc,pf,pg=[np.clip(x,0,None) for x in (pc,pf,pg)]
    for b in lab+['all']:
        k=np.ones(len(X),bool) if b=='all' else (X.ab==b).values
        rows.append(dict(delta=dl,ab=b,people_M=pc[k].sum()/1e6,seen=pf[k].sum()/pc[k].sum(),rebuilt=pg[k].sum()/pc[k].sum()))
    if dl==0:
        m=pc>10; rr=pg[m]/pc[m]; rf=pf[m]/pc[m]
        print('per-crest people ratio (crests with >10 residents):',m.sum(),'first look median',np.median(rf).round(2),'rebuilt median',np.median(rr).round(2),'within25',(abs(rr-1)<=.25).mean().round(3),'over',(rr>1.25).mean().round(3),'under',(rr<0.75).mean().round(3))
        for b in lab:
            kk=m&(X.ab==b).values; q=pg[kk]/pc[kk]; print(' ',b,kk.sum(),'median',np.median(q).round(2),'w25',(abs(q-1)<=.25).mean().round(2),'over',(q>1.25).mean().round(2),'under',(q<.75).mean().round(2))
R=pd.DataFrame(rows); print(R.pivot(index='ab',columns='delta',values='seen').reindex(lab+['all']).round(3)); print(R.pivot(index='ab',columns='delta',values='rebuilt').reindex(lab+['all']).round(3)); print(R[R.ab=='all'].round(3))
R.to_csv('fig/rev/hand_delta.csv',index=False)
