# Full-resolution GOES-16 ACMC clear/cloud snapshots around two example gauges (Aqua-time files)
import netCDF4 as nc, numpy as np, urllib.request, re, datetime as dt, sys
sys.path.insert(0,'code')
B='https://noaa-goes16.s3.amazonaws.com/'
def latlon(d):
  x=np.asarray(d['x'][:],dtype='f8'); y=np.asarray(d['y'][:],dtype='f8'); p=d['goes_imager_projection']
  H=p.perspective_point_height+p.semi_major_axis; a=p.semi_major_axis; b=p.semi_minor_axis; l0=np.deg2rad(p.longitude_of_projection_origin)
  X,Y=np.meshgrid(x,y); A=np.sin(X)**2+np.cos(X)**2*(np.cos(Y)**2+(a/b)**2*np.sin(Y)**2); Bq=-2*H*np.cos(X)*np.cos(Y); C=H**2-a**2
  with np.errstate(invalid='ignore'):
    rs=(-Bq-np.sqrt(Bq*Bq-4*A*C))/(2*A); sx=rs*np.cos(X)*np.cos(Y); sy=-rs*np.sin(X); sz=rs*np.cos(X)*np.sin(Y)
    lat=np.rad2deg(np.arctan((a/b)**2*sz/np.sqrt((H-sx)**2+sy**2))); lon=np.rad2deg(l0-np.arctan(sy/(H-sx)))
  return lat,lon
d0=nc.Dataset('data/acm_sample.nc'); LAT,LON=latlon(d0)
def key(day,hour):
  s=urllib.request.urlopen(B+f'?list-type=2&prefix=ABI-L2-ACMC/{day.year}/{day.timetuple().tm_yday:03d}/{hour:02d}/',timeout=60).read().decode()
  ks=re.findall(r'<Key>([^<]+)</Key>',s); best=min(ks,key=lambda k:abs(int(re.search(r'_s\d{9}(\d{2})',k).group(1))-31)); return best
out={}
for tag,lat0,lon0,dd,days,hour in [('nc',35.309,-80.674,1.6,['2020-11-11','2020-11-12','2020-11-13','2020-11-14'],18),('sd',43.186,-97.635,3.2,['2024-06-22','2024-06-23','2024-06-24','2024-06-25'],19)]:
  m=(np.abs(LAT-lat0)<dd)&(np.abs(LON-lon0)<dd*1.35); r=np.where(m.any(1))[0]; c=np.where(m.any(0))[0]
  sl=(slice(r[0],r[-1]+1),slice(c[0],c[-1]+1)); out[f'{tag}_lat']=LAT[sl]; out[f'{tag}_lon']=LON[sl]
  for day in days:
    k=key(dt.date.fromisoformat(day),hour); buf=urllib.request.urlopen(B+k,timeout=120).read()
    d=nc.Dataset('m',memory=buf); d.set_auto_maskandscale(False); b=np.asarray(d['BCM'][:])[sl].astype('i1'); q=np.asarray(d['DQF'][:])[sl]; d.close()
    b=np.where((b>=0)&((q==0)|(q==2)|(q>=4)),b,-1); out[f'{tag}_{day}']=b; print(tag,day,k.split('/')[-1][:60],(b==0).mean().round(2))
np.savez_compressed('data/case_goes.npz',**out)
