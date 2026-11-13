# Remove duplicate gauges in Caravan: CAMELS (US) basins that HYSETS also carries, and co-located gauges with the same drainage area within a dataset.
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
C=pd.read_parquet('data/global/caravan_matched.parquet')
S=C.groupby(['dataset','basin_id']).agg(lat=('lat','first'),lon=('lon','first'),A=('area_km2','first'),n=('peak_date','size')).reset_index()
S['num']=S.basin_id.str.extract(r'(\d{6,15})$')[0]
t=cKDTree(S[['lat','lon']].values); drop=set()
for i,j in sorted(t.query_pairs(0.005)):
    a,b=S.iloc[i],S.iloc[j]; ar=max(a.A,b.A)/max(min(a.A,b.A),1e-6)
    if {a.dataset,b.dataset}=={'camels','hysets'} and ar<1.25: drop.add(a.basin_id if a.dataset=='camels' else b.basin_id)
    elif a.dataset==b.dataset and ar<1.1: drop.add(a.basin_id if a.n<b.n else b.basin_id)
hy=set(S[S.dataset=='hysets'].num); idd=S[(S.dataset=='camels')&S.num.isin(hy)].basin_id; drop|=set(idd)
print('dropped gauges',len(drop),'of',len(S)); print(S[S.basin_id.isin(drop)].dataset.value_counts().to_dict())
C=C[~C.basin_id.isin(drop)].copy(); C['dataset']=C.dataset.replace({'camels':'namerica','hysets':'namerica'})
C.to_parquet('data/global/caravan_matched_dedup.parquet'); print(C.shape, C.dataset.value_counts().to_dict(), 'gauges',C.basin_id.nunique())
