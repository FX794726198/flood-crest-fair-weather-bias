# Real-image check of the GOES-16 clear-sky proxy: Sentinel-2 L2A scene classification (SCL) within 1 km and 3 km of the gauge, days 0-10 after each crest
import numpy as np, pandas as pd, urllib.request, re, os, sys, time, datetime as dt, mgrs, rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'; os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']='.tif'; os.environ['GDAL_HTTP_MAX_RETRY']='4'; os.environ['GDAL_HTTP_RETRY_DELAY']='2'
BU='https://sentinel-cogs.s3.us-west-2.amazonaws.com/'; PFX='sentinel-s2-l2a-cogs'
T=np.load('data/terra.npy',mmap_mode='r'); D=np.load('data/days.npy'); di={str(d)[:10]:i for i,d in enumerate(D)}
s=pd.read_parquet('data/main_sample.parquet'); lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
s['ab']=pd.cut(s.drain_area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=lab).astype(str)
NPER=int(sys.argv[1]) if len(sys.argv)>1 else 300
smp=pd.concat([d.sample(min(NPER,len(d)),random_state=7) for _,d in s.groupby('ab')]).reset_index(drop=True)
M=mgrs.MGRS(); cache={}
def lst(prefix):
    if prefix in cache: return cache[prefix]
    for k in range(4):
        try:
            x=urllib.request.urlopen(BU+f'?list-type=2&prefix={prefix}&delimiter=/',timeout=60).read().decode(); break
        except Exception: time.sleep(1+k); x=''
    out=re.findall(r'<Prefix>([^<]+_L2A/)</Prefix>',x); cache[prefix]=out; return out
CLEAR={2,4,5,6,7,11}; CLOUD={3,8,9,10}
def one(r):
    try:
        tile=M.toMGRS(r.lat,r.lon,MGRSPrecision=0); z,b,sq=tile[:2],tile[2],tile[3:5]
        d0=dt.date.fromisoformat(r.peak_date); days=[d0+dt.timedelta(o) for o in range(0,11)]
        scenes={}
        for (y,m) in sorted({(d.year,d.month) for d in days}):
            for p in lst(f'{PFX}/{int(z)}/{b}/{sq}/{y}/{m}/'):
                mm=re.search(r'_(\d{8})_(\d+)_L2A/$',p); dd=dt.datetime.strptime(mm.group(1),'%Y%m%d').date()
                if dd in days and (dd not in scenes or int(mm.group(2))>scenes[dd][1]): scenes[dd]=(p,int(mm.group(2)))
        out=[]
        epsg=32600+int(z) if r.lat>=0 else 32700+int(z); tr=Transformer.from_crs(4326,epsg,always_xy=True); x,y=tr.transform(r.lon,r.lat)
        for dd,(p,_) in sorted(scenes.items()):
            with rasterio.open('/vsicurl/'+BU+p+'SCL.tif') as src:
                w=from_bounds(x-3000,y-3000,x+3000,y+3000,src.transform); a=src.read(1,window=w,boundless=True,fill_value=0)
            ny,nx=a.shape; yy,xx=np.mgrid[:ny,:nx]; rr=np.hypot(yy-(ny-1)/2,xx-(nx-1)/2)*20
            res=[]
            for R in (1000,3000):
                v=a[(rr<=R)]; v=v[v>0]; nv=len(v); res+=[nv/(np.pi*(R/20)**2), np.isin(v,list(CLEAR)).mean() if nv else np.nan, np.isin(v,list(CLOUD)).mean() if nv else np.nan]
            k=di.get(dd.isoformat()); g=np.nan if k is None or T[k,r.iy,r.ix]==255 else T[k,r.iy,r.ix]/200
            out.append([r.site_no,r.peak_date,r.drain_area_km2,r.ab,(dd-d0).days,tile]+res+[g])
        return out
    except Exception as e:
        return [[r.site_no,r.peak_date,r.drain_area_km2,r.ab,-99,str(e)[:60]]+[np.nan]*7]
t0=time.time(); rows=[]
with ThreadPoolExecutor(24) as ex:
    for i,o in enumerate(ex.map(one,list(smp.itertuples()))):
        rows+=o
        if i%200==0: print(i,len(rows),round(time.time()-t0),flush=True)
R=pd.DataFrame(rows,columns=['site_no','peak_date','A','ab','day','tile','cov1','clr1','cld1','cov3','clr3','cld3','goes_t'])
R.to_parquet('data/s2/s2_scl.parquet'); print('done',len(R),round(time.time()-t0))
