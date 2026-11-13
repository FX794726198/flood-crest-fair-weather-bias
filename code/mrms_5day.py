import numpy as np, pandas as pd, urllib.request, gzip, eccodes, datetime as dt, time
from concurrent.futures import ThreadPoolExecutor
RC=pd.read_parquet('data/main_sample.parquet'); RC=RC[RC.peak_date>='2020-10-20'].copy(); RC['A']=RC.drain_area_km2
dates=sorted({(dt.date.fromisoformat(p)+dt.timedelta(k)).isoformat() for p in RC.peak_date for k in (-3,-2,-1,0,1)})
print('events',len(RC),'dates',len(dates),flush=True)
ev=RC[['site_no','peak_date','lat','lon','A']].reset_index(drop=True)
rad=np.clip(np.sqrt(ev.A.values/np.pi),2,50)  # km
r0=np.round((54.995-ev.lat.values)/0.01).astype(int); c0=np.round((ev.lon.values+129.995)/0.01).astype(int)
masks=[]
for i in range(len(ev)):
    rr=int(np.ceil(rad[i]/1.11)); cc=int(np.ceil(rad[i]/(1.11*np.cos(np.deg2rad(ev.lat[i])))))
    yy,xx=np.mgrid[-rr:rr+1,-cc:cc+1]; m=(yy*1.11)**2+(xx*1.11*np.cos(np.deg2rad(ev.lat[i])))**2<=rad[i]**2
    masks.append((np.clip(r0[i]+yy[m],0,3499),np.clip(c0[i]+xx[m],0,6999)))
def dl(d):
    ds=d.replace('-','')
    for prod in ['MultiSensor_QPE_24H_Pass2_00.00','MultiSensor_QPE_24H_Pass1_00.00']:
        for k in range(3):
            try: return d,urllib.request.urlopen(f'https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{prod}/{ds}/MRMS_{prod}_{ds}-120000.grib2.gz',timeout=60).read()
            except Exception as e: time.sleep(1)
    return d,None
need={}
for i,p in enumerate(ev.peak_date):
    for k in (-3,-2,-1,0,1): need.setdefault((dt.date.fromisoformat(p)+dt.timedelta(k)).isoformat(),[]).append((i,k+3))
mean=np.full((len(ev),5),np.nan,'f4'); mx=mean.copy(); t0=time.time()
with ThreadPoolExecutor(12) as ex:
    for n,(d,b) in enumerate(ex.map(dl,dates)):
        if b is None: continue
        g=eccodes.codes_new_from_message(gzip.decompress(b)); v=eccodes.codes_get_values(g).reshape(3500,7000).astype('f4'); eccodes.codes_release(g); v[v<0]=np.nan
        for i,k in need[d]:
            x=v[masks[i]]; mean[i,k]=np.nanmean(x); mx[i,k]=np.nanmax(x)
        if n%200==0: print(n,round(time.time()-t0),flush=True)
for k in range(5): ev[f'P{k-3}']=mean[:,k]; ev[f'X{k-3}']=mx[:,k]
ev.to_parquet('data/mrms_5day.parquet'); print(ev.describe().round(1).to_string())
