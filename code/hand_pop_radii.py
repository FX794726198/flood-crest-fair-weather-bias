# HAND area and population (Meta HRSL, 1 arcsec) below each HAND level within 5 km of every gauge in the reconstruction sample
import numpy as np, pandas as pd, rasterio, math, os, sys, time
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'; os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']='.tif,.vrt'; os.environ['GDAL_HTTP_MAX_RETRY']='4'; os.environ['GDAL_HTTP_RETRY_DELAY']='2'
r=pd.read_parquet('data/recon_cv.parquet')
sites=r.groupby('site_no').agg(lat=('lat','first'),lon=('lon','first'),A=('A','first')).reset_index()
N=int(sys.argv[1]) if len(sys.argv)>1 else len(sites); sites=sites.iloc[:N]
DH=np.array([0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15])
POP='/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/hrsl_general-latest.vrt'
def tile(lat,lon):
    la=math.floor(lat); lo=math.floor(lon)
    return f"/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_{'N' if la>=0 else 'S'}{abs(la):02d}_00_{'W' if lo<0 else 'E'}{abs(lo):03d}_00_HAND.tif"
def one(s):
    R=5000; dlat=R/111320; dlon=R/(111320*math.cos(math.radians(s.lat)))
    try:
        with rasterio.open(tile(s.lat,s.lon)) as src:
            w=from_bounds(s.lon-dlon,s.lat-dlat,s.lon+dlon,s.lat+dlat,src.transform); a=src.read(1,window=w,boundless=True,fill_value=-9999).astype('f4')
            wt=src.window_transform(w)
        ny,nx=a.shape; yy,xx=np.mgrid[:ny,:nx]
        lonc=wt.c+(xx+0.5)*wt.a; latc=wt.f+(yy+0.5)*wt.e
        with rasterio.open(POP) as ps:
            pw=from_bounds(s.lon-dlon-0.001,s.lat-dlat-0.001,s.lon+dlon+0.001,s.lat+dlat+0.001,ps.transform); p=ps.read(1,window=pw,boundless=True,fill_value=0); pt=ps.window_transform(pw)
        p=np.where(np.isfinite(p)&(p>0),p,0)
        pr=np.clip(((latc-pt.f)/pt.e).astype(int),0,p.shape[0]-1); pc=np.clip(((lonc-pt.c)/pt.a).astype(int),0,p.shape[1]-1)
        # people per HAND pixel: HRSL count scaled by pixel-area ratio
        ratio=abs(wt.a*wt.e)/abs(pt.a*pt.e); pp=p[pr,pc]*ratio
        rr=np.hypot((latc-s.lat)*111320,(lonc-s.lon)*111320*math.cos(math.radians(s.lat)))
        pa=abs(wt.a*111320*math.cos(math.radians(s.lat))*wt.e*111320)/1e6; out=[s.site_no]
        for RR in (1000,3000,5000):
            m=(rr<=RR)&(a>-1000); v=a[m]; q=pp[m]
            out+=[(v<d).sum()*pa for d in DH]+[(v<=0.01).sum()*pa]+[q[v<d].sum() for d in DH]+[q[v<=0.01].sum(),q.sum()]
        return out
    except Exception as e:
        return [s.site_no]+[np.nan]*(3*(2*len(DH)+3))
t0=time.time()
with ThreadPoolExecutor(16) as ex: res=list(ex.map(one,[s for s in sites.itertuples()]))
H=pd.DataFrame(res,columns=['site_no']+[f'{c}_r{k}' for k in (1,3,5) for c in [f'A{d}' for d in DH]+['A0']+[f'P{d}' for d in DH]+['P0','Ptot']])
H=H.merge(sites,on='site_no'); H.to_parquet('data/hand_pop_radii.parquet')
print(len(H),'nan share',H.Ptot_r5.isna().mean().round(3),'sec',round(time.time()-t0,1))
