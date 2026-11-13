import numpy as np, pandas as pd
T=np.load('data/terra.npy'); A=np.load('data/aqua.npy'); D=np.load('data/days.npy'); clim=np.load('data/clim_month.npy')
di={d:i for i,d in enumerate(D)}
e=pd.read_csv('data/usgs/events.csv',dtype={'site_no':str})
e=e[e.lat.between(24.5,50)&e.lon.between(-125,-66)].copy()
e['iy']=np.floor((e.lat-24.5)/0.1).astype(int); e['ix']=np.floor((e.lon+125)/0.1).astype(int)
OFF=np.arange(-3,11)
t0=np.array([di.get(d,-99999) for d in e.peak_date])
ft=np.full((len(e),len(OFF)),np.nan,'f4'); fa=ft.copy()
for k,o in enumerate(OFF):
    ii=t0+o; ok=(ii>=0)&(ii<len(D))
    vt=T[np.clip(ii,0,len(D)-1),e.iy.values,e.ix.values].astype('f4'); va=A[np.clip(ii,0,len(D)-1),e.iy.values,e.ix.values].astype('f4')
    ft[:,k]=np.where(ok&(vt<255),vt/200,np.nan); fa[:,k]=np.where(ok&(va<255),va/200,np.nan)
m=np.array([int(d[5:7]) for d in e.peak_date])-1
e['clim_t']=clim[0,m,e.iy,e.ix]; e['clim_a']=clim[1,m,e.iy,e.ix]; e['clim_b']=clim[2,m,e.iy,e.ix]
for k,o in enumerate(OFF):
    e[f't{o}']=ft[:,k]; e[f'a{o}']=fa[:,k]
# daily flows -> near-crest days (Q >= 0.8 * max daily Q in window) and Q at first clear look
w=pd.read_parquet('data/usgs/windows.parquet'); w['site_no']=w.site_no.astype(str)
qw=w.pivot_table(index=['site_no','peak_date'],columns='day_off',values='q_cfs')
qw=qw.reindex(columns=range(-15,31))
key=list(zip(e.site_no,e.peak_date)); Q=qw.reindex(key).values  # (n,46) day_off -15..30
qmax=np.nanmax(Q[:,12:19],axis=1)  # max daily within -3..+3
e['qmax_d']=qmax
Qo=Q[:,12:26]  # -3..+10
rel=Qo/qmax[:,None]
for k,o in enumerate(OFF): e[f'q{o}']=rel[:,k]
e.to_parquet('data/matched.parquet'); print(e.shape, np.isfinite(ft).mean(), np.isfinite(rel).mean())
