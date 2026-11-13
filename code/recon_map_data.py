# HAND (GLO-30) and Meta HRSL windows around two example gauges for the reconstruction map
import numpy as np, rasterio, math, os
from rasterio.windows import from_bounds
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'; os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']='.tif,.vrt'
POP='/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/hrsl_general-latest.vrt'
def tile(lat,lon):
    la=math.floor(lat); lo=math.floor(lon)
    return f"/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_N{la:02d}_00_W{abs(lo):03d}_00_HAND.tif"
out={}
for tag,lat,lon in [('nc',35.309,-80.674),('ca',38.438763,-122.70245),('ia',41.971945,-91.667124)]:
    R=6000; dlat=R/111320; dlon=R/(111320*math.cos(math.radians(lat)))
    with rasterio.open(tile(lat,lon)) as s:
        w=from_bounds(lon-dlon,lat-dlat,lon+dlon,lat+dlat,s.transform); a=s.read(1,window=w,boundless=True,fill_value=-9999).astype('f4'); t=s.window_transform(w)
    with rasterio.open(POP) as s:
        w=from_bounds(lon-dlon,lat-dlat,lon+dlon,lat+dlat,s.transform); p=s.read(1,window=w,boundless=True,fill_value=0).astype('f4'); tp=s.window_transform(w)
    out[f'{tag}_hand']=a; out[f'{tag}_ht']=np.array(t)[:6]; out[f'{tag}_pop']=np.where(np.isfinite(p)&(p>0),p,0); out[f'{tag}_pt']=np.array(tp)[:6]
    print(tag,a.shape,p.shape,np.nanmax(a),p.sum())
np.savez_compressed('data/recon_map.npz',**out)
