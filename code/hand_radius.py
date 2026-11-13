import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
r=pd.read_parquet('data/recon_cv.parquet'); H=pd.read_parquet('data/hand_pop_radii.parquet')
DH=np.array([0,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15]); rows=[]
for k in (1,3,5):
    cP=[f'P0_r{k}']+[f'P{d}_r{k}' for d in DH[1:]]
    X=r.merge(H[['site_no']+cP],on='site_no'); X['ab']=pd.cut(X.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab).astype(str); V=X[cP].values.astype(float)
    P=lambda h:np.array([np.interp(a,DH,v) for a,v in zip(np.clip(h,0,15),V)])
    pc=P(X.dh_peak_ft*0.3048)-V[:,0]; pf=P(X.dh_first_ft*0.3048)-V[:,0]; pg=P(X.dpk_gb*0.3048)-V[:,0]
    for b in lab+['all']:
        m=np.ones(len(X),bool) if b=='all' else (X.ab==b).values
        rows.append(dict(radius_km=k,ab=b,people_M=pc[m].sum()/1e6,seen=pf[m].sum()/pc[m].sum(),rebuilt=pg[m].sum()/pc[m].sum()))
R=pd.DataFrame(rows); print(R.pivot(index='ab',columns='radius_km',values='seen').reindex(lab+['all']).round(3)); print(R.pivot(index='ab',columns='radius_km',values='rebuilt').reindex(lab+['all']).round(3)); print(R[R.ab=='all'].round(2))
R.to_csv('fig/rev/hand_radius.csv',index=False)
