# people and area inside the HAND flood footprint at the crest, at the first clear look, and after reconstruction
import numpy as np, pandas as pd
r=pd.read_parquet('data/recon_cv.parquet'); H=pd.read_parquet('data/hand_pop.parquet')
DH=np.array([0,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15])
cA=['A0']+[f'A{d}' for d in DH[1:]]; cP=['P0']+[f'P{d}' for d in DH[1:]]
X=r.merge(H[['site_no']+cA+cP+['Ptot']],on='site_no')
def f(cols,h): 
    V=X[cols].values; hh=np.clip(h*0.3048,0,15); return np.array([np.interp(a,DH,v) for a,v in zip(hh,V)])
for k,c in [('crest','dh_peak_ft'),('first','dh_first_ft'),('gb','dpk_gb')]:
    X[f'P_{k}']=f(cP,X[c].values)-X.P0; X[f'A_{k}']=f(cA,X[c].values)-X.A0   # above the stream-network baseline (HAND = 0), as in Fig. 1
lab=['<30','30–100','100–300','300–1k','1k–3k','3k–10k','>10k']
X['ab']=pd.cut(X.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
g=X.groupby('ab',observed=True)
O=pd.DataFrame({'n':g.size(),'sites':g.site_no.nunique(),'pop_crest_sum':g.P_crest.sum(),'pop_first/crest':g.P_first.sum()/g.P_crest.sum(),'pop_gb/crest':g.P_gb.sum()/g.P_crest.sum(),
   'area_first/crest':g.A_first.sum()/g.A_crest.sum(),'area_gb/crest':g.A_gb.sum()/g.A_crest.sum(),'pop_med_first/crest':g.apply(lambda d:(d.P_first/d.P_crest)[d.P_crest>10].median()),'area_med_first/crest':g.apply(lambda d:(d.A_first/d.A_crest)[d.A_crest>0.01].median()),'pop_med_gb/crest':g.apply(lambda d:(d.P_gb/d.P_crest)[d.P_crest>10].median()),
   'gb_abs_err':g.apply(lambda d:(abs(d.P_gb-d.P_crest).sum()/d.P_crest.sum())),'first_abs_err':g.apply(lambda d:(abs(d.P_first-d.P_crest).sum()/d.P_crest.sum()))})
print(O.round(3).to_string())
tot=X.P_crest.sum(); print('all events: people in crest footprints',round(tot/1e6,2),'M; seen at first clear look',round(X.P_first.sum()/tot,3),'; reconstructed',round(X.P_gb.sum()/tot,3))
sm=X[X.A<300]; print('<300 km2: crest',round(sm.P_crest.sum()/1e6,2),'M, first',round(sm.P_first.sum()/sm.P_crest.sum(),3),'gb',round(sm.P_gb.sum()/sm.P_crest.sum(),3))
X.to_parquet('data/exposure.parquet'); O.to_csv('fig/exposure_by_area.csv')
