import numpy as np, pandas as pd, rasterio, math, os
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'; os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']='.tif'
R=pd.read_parquet('data/iv_capture.parquet')
DH=np.array([0,0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6,8,10,12,15])
def tile(lat,lon):
    la=math.floor(lat); lo=math.floor(lon)
    return f"/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_N{la:02d}_00_W{abs(lo):03d}_00_HAND.tif"
def hyp(a):
    site,lat,lon=a; Rr=3000; dlat=Rr/111320; dlon=Rr/(111320*math.cos(math.radians(lat)))
    try:
        with rasterio.open(tile(lat,lon)) as src:
            w=from_bounds(lon-dlon,lat-dlat,lon+dlon,lat+dlat,src.transform); x=src.read(1,window=w,boundless=True,fill_value=-9999).astype('f4')
        yy,xx=np.mgrid[:x.shape[0],:x.shape[1]]; cy,cx=(x.shape[0]-1)/2,(x.shape[1]-1)/2
        rr=np.hypot((yy-cy)/cy,(xx-cx)/cx); v=x[(rr<=1)&(x>-1000)]
        return [site]+[(v<=d+1e-3).mean() for d in DH]
    except Exception as e: return [site]+[np.nan]*len(DH)
U=R[['site_no','lat','lon']].drop_duplicates('site_no').values.tolist()
with ThreadPoolExecutor(12) as ex: Hh=pd.DataFrame(list(ex.map(hyp,U)),columns=['site_no']+[f'h{d}' for d in DH])
M=R.merge(Hh,on='site_no'); cols=[f'h{d}' for d in DH]
def frac(row,col):
    a=row[cols].values.astype(float); full=np.interp(row.depth_m,DH,a)-a[0]
    return (np.interp(row[col]*row.depth_m,DH,a)-a[0])/full if full>1e-4 else np.nan
for c in ['cloudfree_max','clim_expected','observed_max']: M['ext_'+c]=M.apply(lambda r:frac(r,c),axis=1)
M['ab']=pd.cut(M.A,[0,30,100,300,1000,3000,10000,1e7],labels=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k'])
g=M.groupby('ab',observed=True); print(g[['ext_cloudfree_max','ext_clim_expected','ext_observed_max']].median().round(2).to_string()); print(g[['ext_cloudfree_max','ext_clim_expected','ext_observed_max']].mean().round(2).to_string())
M.to_parquet('data/iv_capture_ext.parquet')
