# 15-minute capture with a cloud null that keeps persistence: cloud sequences of the same cell from the same calendar dates (shifted by 0, +-7, +-14 days) in the other years
import numpy as np, pandas as pd, datetime as dt
T=np.load('data/terra.npy',mmap_mode='r'); A=np.load('data/aqua.npy',mmap_mode='r'); D=np.load('data/days.npy'); clim=np.load('data/clim_month.npy')
di={d:i for i,d in enumerate(D)}
v=pd.read_parquet('data/usgs/iv.parquet'); v=v[v.param=='00065'].dropna(subset=['value'])
e=pd.read_csv('data/usgs/events.csv',dtype={'site_no':str})
X=pd.read_parquet('data/iv_capture_ext.parquet'); keep=set(zip(X.site_no,X.peak_date))
hy=X.set_index(['site_no','peak_date'])
DH=np.array([0,0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6,8,10,12,15]); hcols=[f'h{d}' for d in DH]
def ext(hrow,depth,f):
    a=hrow[hcols].values.astype(float); full=np.interp(depth,DH,a)-a[0]
    return (np.interp(f*depth,DH,a)-a[0])/full if full>1e-4 else np.nan
SH=[-14,-7,0,7,14]; rows=[]
for (site,pdate),g in v.groupby(['site_no','peak_date']):
    if (site,pdate) not in keep: continue
    r=e[(e.site_no==site)&(e.peak_date==pdate)].iloc[0]; g=g.sort_values('time_utc')
    t=(g.time_utc-pd.Timestamp(pdate,tz='UTC')).dt.total_seconds().values/3600; h=g.value.values
    ic=np.argmax(h); tc=t[ic]; hc=h[ic]; pre=(t<tc)&(t>=tc-72); hb=np.min(h[pre]) if pre.sum()>4 else np.min(h[:ic+1])
    iy=int((r.lat-24.5)/0.1); ix=int((r.lon+125)/0.1); d0=dt.date.fromisoformat(pdate)
    looks=[]
    for o in range(-3,11):
        for sat,loc,cube in (('T',10.5,T),('A',13.5,A)):
            tu=o*24+loc-r.lon/15
            if tu<t[0] or tu>t[-1]: continue
            hv=np.interp(tu,t,h); k=di.get((d0+dt.timedelta(o)).isoformat())
            f=np.nan if k is None or cube[k,iy,ix]==255 else cube[k,iy,ix]/200
            alt=[]
            for Y in range(2018,2025):
                if Y==d0.year: continue
                try: base=d0.replace(year=Y)
                except ValueError: base=d0.replace(year=Y,day=28)
                for sh in SH:
                    kk=di.get((base+dt.timedelta(o+sh)).isoformat()); alt.append(np.nan if kk is None or cube[kk,iy,ix]==255 else cube[kk,iy,ix]/200)
            looks.append([tu,o,(hv-hb)/(hc-hb),f]+alt)
    L=np.array(looks,dtype=float); df=L[:,2]; obs=L[:,3]; alt=L[:,4:]
    nul=[]
    for j in range(alt.shape[1]):
        c=alt[:,j]
        if np.isnan(c).mean()>0.5: continue
        hit=c>=0.5; nul.append(df[hit].max() if hit.any() else 0.0)
    hrow=hy.loc[(site,pdate)]; dep=hrow.depth_m
    cf=df.max(); ob=df[obs>=0.5].max() if (obs>=0.5).any() else 0.0
    before=(L[:,0]<tc)&(obs>=0.5); after=(L[:,0]>=tc)&(obs>=0.5); day0=(L[:,1]>=0)&(obs>=0.5)
    rows.append(dict(site_no=site,peak_date=pdate,A=r.drain_area_km2,n_null=len(nul),df_null=np.mean(nul) if nul else np.nan,
        ext_null=np.mean([ext(hrow,dep,x) for x in nul]) if nul else np.nan,
        df_before=df[before][-1] if before.any() else np.nan, df_after=df[after][0] if after.any() else np.nan,
        df_first_day0=df[day0][0] if day0.any() else np.nan, first_day0_before_crest=(L[day0][0,0]<tc) if day0.any() else np.nan,
        t_crest_local=(tc+r.lon/15)%24))
R=pd.DataFrame(rows).merge(X[['site_no','peak_date','ext_cloudfree_max','ext_clim_expected','ext_observed_max','cloudfree_max','clim_expected','observed_max']],on=['site_no','peak_date'])
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; R['ab']=pd.cut(R.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
g=R.groupby('ab',observed=True)
print(g[['ext_cloudfree_max','ext_clim_expected','ext_null','ext_observed_max']].median().round(2).to_string())
print(g[['ext_cloudfree_max','ext_clim_expected','ext_null','ext_observed_max']].mean().round(2).to_string())
print('draws per crest',R.n_null.median()); print(g[['df_before','df_after','df_first_day0']].median().round(2).to_string())
print('share first look on/after crest day that precedes the crest', R.first_day0_before_crest.mean().round(3)); print(g.first_day0_before_crest.mean().round(2).to_dict())
R.to_parquet('data/iv_null.parquet')
# daily-mean approximation: depth fraction at first clear look from daily flow + rating vs 15-minute stage at the overpass
S=pd.read_parquet('data/stage_first.parquet')[['site_no','peak_date','depth_frac']]
Z=R.merge(S,on=['site_no','peak_date']); Z['diff']=Z.depth_frac-Z.df_first_day0
print('n',len(Z)); print(Z.groupby('ab',observed=True).agg(n=('diff','size'),daily=('depth_frac','median'),iv=('df_first_day0','median'),mad=('diff',lambda x:x.abs().median())).round(2).to_string())
