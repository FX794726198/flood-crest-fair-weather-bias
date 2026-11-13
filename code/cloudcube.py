# Daily clear-sky fraction at Terra (10:30) and Aqua (13:30) local solar time on a 0.1 deg CONUS grid from GOES-16 ABI-L2-ACMC
import netCDF4 as nc, numpy as np, urllib.request, re, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
B='https://noaa-goes16.s3.amazonaws.com/'
LAT0,LON0,RES,NY,NX=24.5,-125.0,0.1,255,590
HRS=np.arange(15,22)
def latlon(d):
  x=np.asarray(d['x'][:],dtype='f8'); y=np.asarray(d['y'][:],dtype='f8'); p=d['goes_imager_projection']
  H=p.perspective_point_height+p.semi_major_axis; a=p.semi_major_axis; b=p.semi_minor_axis; l0=np.deg2rad(p.longitude_of_projection_origin)
  X,Y=np.meshgrid(x,y); A=np.sin(X)**2+np.cos(X)**2*(np.cos(Y)**2+(a/b)**2*np.sin(Y)**2); Bq=-2*H*np.cos(X)*np.cos(Y); C=H**2-a**2
  with np.errstate(invalid='ignore'):
    rs=(-Bq-np.sqrt(Bq*Bq-4*A*C))/(2*A); sx=rs*np.cos(X)*np.cos(Y); sy=-rs*np.sin(X); sz=rs*np.cos(X)*np.sin(Y)
    lat=np.rad2deg(np.arctan((a/b)**2*sz/np.sqrt((H-sx)**2+sy**2))); lon=np.rad2deg(l0-np.arctan(sy/(H-sx)))
  return lat,lon
d0=nc.Dataset('data/acm_sample.nc'); lat,lon=latlon(d0)
iy=np.floor((lat-LAT0)/RES); ix=np.floor((lon-LON0)/RES)
ok=np.isfinite(iy)&np.isfinite(ix)&(iy>=0)&(iy<NY)&(ix>=0)&(ix<NX)
cell=np.where(ok,iy*NX+ix,-1).astype(np.int64).ravel(); okr=cell>=0; cellv=cell[okr]
npix=np.bincount(cellv,minlength=NY*NX)
lonc=LON0+RES*(np.arange(NX)+0.5)
def pick(loc): # nearest hour file (HH:31) per column
  utc=loc-lonc/15.0; return np.clip(np.round(utc-0.52).astype(int)-HRS[0],0,len(HRS)-1)
PT,PA=pick(10.5),pick(13.5)
def get(url):
  for k in range(4):
    try: return urllib.request.urlopen(url,timeout=60).read()
    except Exception as e: time.sleep(1+k)
  return None
def keys_for(day):
  s=get(B+f'?list-type=2&prefix=ABI-L2-ACMC/{day.year}/{day.timetuple().tm_yday:03d}/')
  if s is None: return {}
  ks=re.findall(r'<Key>([^<]+)</Key>',s.decode()); out={}
  for h in HRS:
    best=None
    for k in ks:
      m=re.search(r'_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})',k)
      if m and int(m.group(3))==h:
        dm=abs(int(m.group(4))+int(m.group(5))/60-31.5)
        if best is None or dm<best[0]: best=(dm,k)
    if best: out[h]=best[1]
  return out
def frac(buf):
  d=nc.Dataset('m',memory=buf); d.set_auto_maskandscale(False); b=np.asarray(d['BCM'][:]).ravel()[okr]; q=np.asarray(d['DQF'][:]).ravel()[okr]; d.close()
  v=(b>=0)&((q==0)|(q==2)|(q>=4)); nv=np.bincount(cellv[v],minlength=NY*NX); nc_=np.bincount(cellv[v&(b==0)],minlength=NY*NX)
  f=np.full(NY*NX,255,np.uint8); g=(npix>0)&(nv>=0.5*npix); f[g]=np.round(200*nc_[g]/nv[g]).astype(np.uint8); return f.reshape(NY,NX)
def run(year,d1=None,d2=None):
  days=[dt.date(year,1,1)+dt.timedelta(i) for i in range((dt.date(year+1,1,1)-dt.date(year,1,1)).days)]
  if d1 is not None: days=days[d1:d2]
  T=np.full((len(days),NY,NX),255,np.uint8); A=T.copy(); t0=time.time()
  with ThreadPoolExecutor(16) as ex:
    K=list(ex.map(keys_for,days))
    jobs=[(i,h,k) for i,kk in enumerate(K) for h,k in kk.items()]
    bufs=ex.map(lambda j:get(B+j[2]),jobs)
    cube={}
    for (i,h,k),buf in zip(jobs,bufs):
      if buf is None: continue
      try: cube[(i,h)]=frac(buf)
      except Exception as e: print('bad',k,e)
      if len([1 for (ii,hh) in cube if ii==i])==len(K[i]):
        st=np.stack([cube.get((i,hh),np.full((NY,NX),255,np.uint8)) for hh in HRS])
        T[i]=np.take_along_axis(st,PT[None,None,:].repeat(NY,1),0)[0]; A[i]=np.take_along_axis(st,PA[None,None,:].repeat(NY,1),0)[0]
        for hh in HRS: cube.pop((i,hh),None)
  print(year,len(days),'days',len(jobs),'files',round(time.time()-t0),'s',flush=True)
  return days,T,A
if __name__=='__main__':
  y=int(sys.argv[1])
  if len(sys.argv)>2:
    days,T,A=run(y,0,int(sys.argv[2])); print('terra valid',(T<255).mean(),'clear',T[T<255].mean()/200,'aqua clear',A[A<255].mean()/200)
  else:
    days,T,A=run(y); np.savez_compressed(f'data/clear_{y}.npz',terra=T,aqua=A,days=np.array([str(x) for x in days]))
