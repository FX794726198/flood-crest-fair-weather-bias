import numpy as np, pandas as pd, datetime as dt
T=np.load('data/terra.npy',mmap_mode='r'); A=np.load('data/aqua.npy',mmap_mode='r'); D=np.load('data/days.npy'); clim=np.load('data/clim_month.npy')
di={d:i for i,d in enumerate(D)}
v=pd.read_parquet('data/usgs/iv.parquet'); v=v[v.param=='00065'].dropna(subset=['value'])
e=pd.read_csv('data/usgs/events.csv',dtype={'site_no':str})
rng=np.random.default_rng(0); rows=[]
for (site,pdate),g in v.groupby(['site_no','peak_date']):
    r=e[(e.site_no==site)&(e.peak_date==pdate)]
    if r.empty: continue
    r=r.iloc[0]; g=g.sort_values('time_utc'); t=(g.time_utc-pd.Timestamp(pdate,tz='UTC')).dt.total_seconds().values/3600; h=g.value.values
    if len(h)<50: continue
    ic=np.argmax(h); tc=t[ic]; hc=h[ic]; pre=(t<tc)&(t>=tc-72); hb=np.min(h[pre]) if pre.sum()>4 else np.min(h[:ic+1])
    if hc-hb<0.3: continue
    iy=int((r.lat-24.5)/0.1); ix=int((r.lon+125)/0.1); m=int(pdate[5:7])-1
    looks=[]
    for o in range(-3,11):
        day=(dt.date.fromisoformat(pdate)+dt.timedelta(o)).isoformat(); k=di.get(day)
        for sat,loc,cube,ci in (('T',10.5,T,0),('A',13.5,A,1)):
            tu=o*24+loc-r.lon/15  # hours after peak_date 00 UTC
            if tu<t[0] or tu>t[-1]: continue
            hv=np.interp(tu,t,h); f=np.nan if k is None or cube[k,iy,ix]==255 else cube[k,iy,ix]/200
            looks.append((tu,sat,(hv-hb)/(hc-hb),f,clim[ci,m,iy,ix]))
    L=pd.DataFrame(looks,columns=['tu','sat','df','clear','clim'])
    if L.empty: continue
    allmax=L.df.max()
    obs=L[L.clear>=0.5]; obsmax=obs.df.max() if len(obs) else 0.0
    # expected under climatological (random) cloud: each look clear with prob clim, independent
    sims=[]
    for _ in range(300):
        c=rng.random(len(L))<L.clim.values; sims.append(L.df[c].max() if c.any() else 0.0)
    after=L[(L.tu>=tc)&(L.clear>=0.5)]
    rows.append(dict(site_no=site,peak_date=pdate,A=r.drain_area_km2,lat=r.lat,lon=r.lon,depth_m=(hc-hb)*0.3048,t_crest_h=tc,
        cloudfree_max=allmax,observed_max=obsmax,clim_expected=np.mean(sims),first_after_lag_h=(after.tu.iloc[0]-tc) if len(after) else np.nan,
        first_after_df=after.df.iloc[0] if len(after) else np.nan,crest_overpass_gap_h=np.min(np.abs(L.tu-tc))))
R=pd.DataFrame(rows); R.to_parquet('data/iv_capture.parquet')
R['ab']=pd.cut(R.A,[0,30,100,300,1000,3000,10000,1e7],labels=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k'])
g=R.groupby('ab',observed=True)
print(pd.DataFrame({'n':g.size(),'depth_m':g.depth_m.median(),'cloudfree_max':g.cloudfree_max.median(),'clim_expected':g.clim_expected.median(),'observed_max':g.observed_max.median(),
  'obs_mean':g.observed_max.mean(),'clim_mean':g.clim_expected.mean(),'cf_mean':g.cloudfree_max.mean(),'lag_h_med':g.first_after_lag_h.median(),'df_first':g.first_after_df.median()}).round(2).to_string())
