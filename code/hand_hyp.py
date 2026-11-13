# HAND hypsometry (area of pixels with HAND < dh) in a 5 km radius around each sampled gauge
import numpy as np, pandas as pd, rasterio, math, os
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'; os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']='.tif'
s=pd.read_parquet('data/main_sample.parquet')
s=s[s.q_at_first.notna()]
rng=np.random.default_rng(1)
sites=s.groupby('site_no').agg(lat=('lat','first'),lon=('lon','first'),A=('drain_area_km2','first'),abin=('abin','first')).reset_index()
sites['b2']=pd.cut(sites.A,[0,30,100,300,1000,3000,10000,1e7],labels=False); pick=pd.concat([d.sample(min(len(d),150),random_state=1) for _,d in sites.groupby('b2')])
DH=np.array([0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15])
def tile(lat,lon):
    la=math.floor(lat); lo=math.floor(lon)
    return f"/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_{'N' if la>=0 else 'S'}{abs(la):02d}_00_{'W' if lo<0 else 'E'}{abs(lo):03d}_00_HAND.tif"
def hyp(r):
    R=5000; dlat=R/111320; dlon=R/(111320*math.cos(math.radians(r.lat)))
    try:
        with rasterio.open(tile(r.lat,r.lon)) as src:
            w=from_bounds(r.lon-dlon,r.lat-dlat,r.lon+dlon,r.lat+dlat,src.transform); a=src.read(1,window=w,boundless=True,fill_value=-9999).astype('f4')
        yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]; cy,cx=(a.shape[0]-1)/2,(a.shape[1]-1)/2
        rr=np.hypot((yy-cy)*dlat*2/a.shape[0]*111320,(xx-cx)*dlon*2/a.shape[1]*111320*math.cos(math.radians(r.lat)))
        v=a[(rr<=R)&(a>-1000)]
        pa=(dlat*2/a.shape[0]*111320)*(dlon*2/a.shape[1]*111320*math.cos(math.radians(r.lat)))
        return [r.site_no]+[(v<d).sum()*pa/1e6 for d in DH]+[(v<=0.01).sum()*pa/1e6]
    except Exception as e:
        return [r.site_no]+[np.nan]*(len(DH)+1)
with ThreadPoolExecutor(12) as ex: res=list(ex.map(hyp,[r for r in pick.itertuples()]))
H=pd.DataFrame(res,columns=['site_no']+[f'A{d}' for d in DH]+['A0'])
H=H.merge(sites[['site_no','lat','lon','A']],on='site_no',how='left'); H['abin']=pd.cut(H.A,[0,30,100,300,1000,3000,10000,1e7],labels=['<30','30–100','100–300','300–1k','1k–3k','3k–10k','>10k']); H.to_csv('data/hand_hyp.csv',index=False); print(H.shape, H.isna().any(axis=1).mean()); print(H.groupby('abin',observed=True)[['A0','A1','A3','A6']].median().round(2))
