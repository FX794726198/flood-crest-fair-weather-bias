# Sentinel-2 scene classes within 1 km of the gauge: cloud-free share by class, and the comparison with GOES-16 for a strict surface-only definition
import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
N=pd.read_parquet('data/s2/s2_scl_classes.parquet'); O=pd.read_parquet('data/s2/s2_scl.parquet')
print('rows new',len(N),'old',len(O),'errors new',(N.day==-99).sum(),'old',(O.day==-99).sum())
f=lambda R:R[(R.day>=0)&(R.cov1>=0.9)&R.goes_t.notna()].copy()
N=f(N); O=f(O); print('valid scenes new',len(N),'crests',N[['site_no','peak_date']].drop_duplicates().shape[0],'old',len(O),O[['site_no','peak_date']].drop_duplicates().shape[0])
M=N.merge(O[['site_no','peak_date','day','clr1']],on=['site_no','peak_date','day'],suffixes=('','_old'),how='inner'); print('matched scenes',len(M),'max |clr1 diff|',np.abs(M.clr1-M.clr1_old).max())
NAME={1:'saturated',2:'dark area',3:'cloud shadow',4:'vegetation',5:'bare soil',6:'water',7:'unclassified',8:'cloud medium',9:'cloud high',10:'thin cirrus',11:'snow'}
print('mean class share within 1 km over all scenes'); print(pd.Series({NAME[c]:N[f'c{c}'].mean() for c in range(1,12)}).round(3).to_string())
N['surf']=N.c4+N.c5+N.c6; N['other']=N.c2+N.c7+N.c11
g=N.goes_t>=0.5
for nm,col in [('cloud-free (six classes)','clr1'),('surface only (vegetation, bare soil, water)','surf')]:
    s=N[col]>=0.5
    print(nm,'| P(S2 | GOES clear)',round(s[g].mean(),3),(s&g).sum(),g.sum(),'| P(S2 | GOES cloudy)',round(s[~g].mean(),3),(s&~g).sum(),(~g).sum(),'| agreement',round((s==g).mean(),3),'by class',N.assign(a=(s==g)).groupby('ab').a.mean().reindex(lab).round(3).to_dict())
    N['early']=N.day<=1
    T=N.assign(s=s).groupby(['ab','early']).s.mean().unstack().reindex(lab).round(2); print(T.to_string())
    F=N.assign(s=s).sort_values('day').groupby(['site_no','peak_date']).apply(lambda d:pd.Series({'fs':d.day[d.s].min(),'fg':d.day[d.goes_t>=0.5].min()}))
    B=F.dropna(); print('  crests',len(F),'both',len(B),'S2 only',(F.fs.notna()&F.fg.isna()).sum(),'GOES only',(F.fs.isna()&F.fg.notna()).sum(),'neither',(F.fs.isna()&F.fg.isna()).sum(),'same first date',round((B.fs==B.fg).mean(),3))
cf=N[N.clr1>=0.5]; print('in cloud-free scenes: share of pixels that are snow/dark/unclassified, mean',round((cf.other/cf.clr1).mean(),3),'scenes where these are over half of the cloud-free pixels',round(((cf.other/cf.clr1)>0.5).mean(),3), 'snow-dominated', round((cf.c11/cf.clr1>0.5).mean(),3))
print('scenes cloud-free by six classes but not by surface classes',((N.clr1>=0.5)&(N.surf<0.5)).sum(),'of',(N.clr1>=0.5).sum())
print('winter share of those', (pd.to_datetime(N[(N.clr1>=0.5)&(N.surf<0.5)].peak_date).dt.month.isin([12,1,2,3])).mean().round(2))
N.to_parquet('data/s2/s2_scl_classes_valid.parquet')
