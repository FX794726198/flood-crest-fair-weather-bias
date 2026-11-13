# Independent check of the HAND footprint: Sentinel-2 open water (MNDWI) in clear scenes within 6 days after the crest, minus water in a clear pre-event scene,
# against the HAND footprint at the gauge depth of that day, within 3 km of the gauge
import numpy as np, pandas as pd, urllib.request, re, os, sys, time, datetime as dt, mgrs, rasterio, math
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'; os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']='.tif'; os.environ['GDAL_HTTP_MAX_RETRY']='4'; os.environ['GDAL_HTTP_RETRY_DELAY']='2'
BU='https://sentinel-cogs.s3.us-west-2.amazonaws.com/'; PFX='sentinel-s2-l2a-cogs'; FT=0.3048; R=3000
D=pd.read_parquet('data/rev_depth_days.parquet'); D=D[(D.dpk*FT>=0.6)].sample(frac=1,random_state=11).reset_index(drop=True)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; D['ab']=pd.cut(D.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab).astype(str)
NMAX=int(sys.argv[1]) if len(sys.argv)>1 else 3000
D=pd.concat([d.head(NMAX//7+1) for _,d in D.groupby('ab')]).reset_index(drop=True)
M=mgrs.MGRS(); cache={}
def lst(prefix):
    if prefix in cache: return cache[prefix]
    x=''
    for k in range(4):
        try: x=urllib.request.urlopen(BU+f'?list-type=2&prefix={prefix}&delimiter=/',timeout=60).read().decode(); break
        except Exception: time.sleep(1+k)
    out=re.findall(r'<Prefix>([^<]+_L2A/)</Prefix>',x); cache[prefix]=out; return out
def scenes(tile,days):
    z,b,sq=tile[:2],tile[2],tile[3:5]; out={}
    for (y,m) in sorted({(d.year,d.month) for d in days}):
        for p in lst(f'{PFX}/{int(z)}/{b}/{sq}/{y}/{m}/'):
            mm=re.search(r'_(\d{8})_(\d+)_L2A/$',p); dd=dt.datetime.strptime(mm.group(1),'%Y%m%d').date()
            if dd in days and (dd not in out or int(mm.group(2))>out[dd][1]): out[dd]=(p,int(mm.group(2)))
    return {k:v[0] for k,v in out.items()}
def rd(p,band,x,y,shape=None):
    with rasterio.open('/vsicurl/'+BU+p+band+'.tif') as src:
        w=from_bounds(x-R,y-R,x+R,y+R,src.transform)
        return src.read(1,window=w,boundless=True,fill_value=0,out_shape=shape,resampling=Resampling.average if shape else Resampling.nearest).astype('f4')
CLEAR=np.array([2,4,5,6,7,11])
def hand_tile(lat,lon):
    la=math.floor(lat); lo=math.floor(lon)
    return f"/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/Copernicus_DSM_COG_10_{'N' if la>=0 else 'S'}{abs(la):02d}_00_{'W' if lo<0 else 'E'}{abs(lo):03d}_00_HAND.tif"
def one(r):
    try:
        tile=M.toMGRS(r.lat,r.lon,MGRSPrecision=0); z=int(tile[:2]); epsg=32600+z
        tr=Transformer.from_crs(4326,epsg,always_xy=True); x,y=tr.transform(r.lon,r.lat)
        d0=dt.date.fromisoformat(r.peak_date)
        ev=scenes(tile,[d0+dt.timedelta(o) for o in range(0,7)])
        best=None
        for dd,p in sorted(ev.items()):
            o=(dd-d0).days; dep=getattr(r,f'd{o}')*FT
            if not np.isfinite(dep) or dep<0.3: continue
            scl=rd(p,'SCL',x,y); ny,nx=scl.shape; yy,xx=np.mgrid[:ny,:nx]; cir=np.hypot(yy-(ny-1)/2,xx-(nx-1)/2)*20<=R
            v=scl[cir]; v=v[v>0]
            if len(v)<0.9*cir.sum(): continue
            cs=np.isin(v,CLEAR).mean()
            if cs>=0.9: best=(dd,p,o,dep,cs); break
        if best is None: return None
        ref=scenes(tile,[d0-dt.timedelta(k) for k in range(5,46)]); rbest=None
        for dd,p in sorted(ref.items(),reverse=True):
            scl=rd(p,'SCL',x,y); v=scl[cir]; v=v[v>0]
            if len(v)<0.9*cir.sum(): continue
            cs=np.isin(v,CLEAR).mean()
            if cs>=0.95: rbest=(dd,p,cs); break
        if rbest is None: return None
        def water(p):
            s=rd(p,'SCL',x,y); g=rd(p,'B03',x,y,shape=s.shape); sw=rd(p,'B11',x,y); m=(g+sw)>0
            nd=np.where(m,(g-sw)/np.maximum(g+sw,1),np.nan); return nd>0, np.isin(s,CLEAR)&(s>0)
        we,ce=water(best[1]); wr,cr=water(rbest[1])
        # HAND on the S2 grid
        tb=Transformer.from_crs(epsg,4326,always_xy=True)
        with rasterio.open(scenes_path:= '/vsicurl/'+BU+best[1]+'SCL.tif') as src:
            w=from_bounds(x-R,y-R,x+R,y+R,src.transform); wt=src.window_transform(w)
        ny,nx=we.shape; jj,ii=np.meshgrid(np.arange(nx),np.arange(ny)); X=wt.c+(jj+0.5)*wt.a; Y=wt.f+(ii+0.5)*wt.e
        lo,la=tb.transform(X,Y)
        with rasterio.open(hand_tile(r.lat,r.lon)) as hs:
            hw=from_bounds(lo.min()-0.002,la.min()-0.002,lo.max()+0.002,la.max()+0.002,hs.transform); hh=hs.read(1,window=hw,boundless=True,fill_value=-9999).astype('f4'); ht=hs.window_transform(hw)
        hr=np.clip(((la-ht.f)/ht.e).astype(int),0,hh.shape[0]-1); hc=np.clip(((lo-ht.c)/ht.a).astype(int),0,hh.shape[1]-1); HA=hh[hr,hc]
        dom=cir&ce&cr&(HA>-1000)&(HA<=15)
        obs=we&~wr&dom; refw=wr&dom
        dep=best[3]; out=dict(site_no=r.site_no,peak_date=r.peak_date,A=r.A,ab=r.ab,day=best[2],depth_m=dep,crest_m=r.dpk*FT,clear_evt=best[4],ref_day=(rbest[0]-d0).days,
            dom_km2=dom.sum()*4e-4,obs_km2=obs.sum()*4e-4,refw_km2=refw.sum()*4e-4,evtw_km2=(we&dom).sum()*4e-4)
        for dl in np.arange(-3,3.01,0.25):
            pr=dom&(HA>0.01)&(HA<dep+dl)&~wr
            out[f'pred_{dl:+.2f}']=pr.sum()*4e-4
            if abs(dl)<1e-6:
                tp=(pr&obs).sum(); fp=(pr&~obs).sum(); fn=(~pr&obs).sum(); out.update(csi=tp/max(tp+fp+fn,1),hit=tp/max(tp+fn,1),far=fp/max(tp+fp,1))
        # observed water vs HAND: median HAND of newly flooded pixels and the HAND level holding 90% of them
        hv=HA[obs]; out['hand_obs_p50']=np.median(hv) if len(hv) else np.nan; out['hand_obs_p90']=np.percentile(hv,90) if len(hv) else np.nan
        return out
    except Exception as e:
        return dict(site_no=r.site_no,peak_date=r.peak_date,err=str(e)[:80])
t0=time.time(); rows=[]
with ThreadPoolExecutor(24) as ex:
    for i,o in enumerate(ex.map(one,list(D.itertuples()))):
        if o: rows.append(o)
        if i%200==0: print(i,len(rows),round(time.time()-t0),flush=True)
W=pd.DataFrame(rows); W.to_parquet('data/s2/s2_water.parquet'); print('done',len(W),round(time.time()-t0))
