# Observing-system simulation on 392 gauged 15-min crests: how much crest flood extent (and how many people) each
# sampling scheme would capture. Optical looks use the actual GOES-16 clear/cloud state at their local overpass time;
# SAR looks are cloud-free at 06:00/18:00 local solar time. Extent follows each gauge's HAND hypsometry (3 km);
# people follow HAND x HRSL (5 km, above the HAND=0 network).
import numpy as np, pandas as pd, datetime as dt, itertools
T=np.load('data/terra.npy',mmap_mode='r'); A=np.load('data/aqua.npy',mmap_mode='r'); D=np.load('data/days.npy',allow_pickle=True)
di={str(d)[:10]:i for i,d in enumerate(D)}
v=pd.read_parquet('data/usgs/iv.parquet'); v=v[v.param=='00065'].dropna(subset=['value'])
E=pd.read_parquet('data/iv_capture_ext.parquet'); E=E[np.isfinite(E.ext_cloudfree_max)].reset_index(drop=True)
HP=pd.read_parquet('data/hand_pop.parquet').set_index('site_no')
DH=np.array([0,0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6,8,10,12,15]); hc=[f'h{d}' for d in DH]
DP=np.array([0,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,15]); pcol=['P0']+[f'P{d}' for d in DP[1:]]
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
RV=np.array([1,2,3,4,6,9,12,18,24,36,48,72,96,120,144,192,240,288])  # revisit (h) for the cloud-free sweep
NPH=96
rng=np.random.default_rng(1)
def clear(cube,k,iy,ix):
    if k is None: return False
    x=cube[k,iy,ix]; return x!=255 and x/200>=0.5
rows=[]; sweep=[]; dur=[]; ex_series={}
for i,r in E.iterrows():
    g=v[(v.site_no==r.site_no)&(v.peak_date==r.peak_date)].sort_values('time_utc')
    t=(g.time_utc-pd.Timestamp(r.peak_date,tz='UTC')).dt.total_seconds().values/3600; h=g.value.values
    ic=np.argmax(h); tc=t[ic]; pre=(t<tc)&(t>=tc-72); hb=np.min(h[pre]) if pre.sum()>4 else np.min(h[:ic+1])
    tg=np.arange(max(t[0],tc-48),min(t[-1],tc+240),0.25); y=np.clip((np.interp(tg,t,h)-hb)/(h[ic]-hb),0,1)
    a=r[hc].values.astype(float); full=np.interp(r.depth_m,DH,a)-a[0]
    ext=lambda yy:(np.interp(np.asarray(yy)*r.depth_m,DH,a)-a[0])/full
    if r.site_no in HP.index:
        P=HP.loc[r.site_no,pcol].values.astype(float); pc=np.interp(min(r.depth_m,15),DP,P)-P[0]
        pop=lambda yy:np.interp(np.clip(np.asarray(yy)*r.depth_m,0,15),DP,P)-P[0]
    else: pc=np.nan; pop=lambda yy:np.full(np.shape(yy),np.nan)
    ey=ext(y); k0=np.argmin(np.abs(tg-tc))
    def run(th):  # contiguous time around crest with extent >= th
        lo=k0; hi=k0
        while lo>0 and ey[lo-1]>=th: lo-=1
        while hi<len(ey)-1 and ey[hi+1]>=th: hi+=1
        return (hi-lo+1)*0.25
    dur.append(dict(site_no=r.site_no,peak_date=r.peak_date,A=r.A,D50=run(0.5),D80=run(0.8),D90=run(0.9)))
    yat=lambda tt:np.interp(tt,tg,y,left=np.nan,right=np.nan)
    # cloud-free revisit sweep, continuous phase
    for rv in RV:
        ph=(np.arange(NPH)+0.5)/NPH*rv; best=[]
        for p in ph:
            tt=tg[0]+p+np.arange(0,tg[-1]-tg[0],rv); best.append(np.nanmax(yat(tt)) if len(tt) else 0.0)
        best=np.nan_to_num(np.array(best))
        sweep.append(dict(site_no=r.site_no,peak_date=r.peak_date,A=r.A,rv=rv,ext=ext(best).mean(),p80=(ext(best)>=0.8).mean(),pop=pop(best).mean(),pc=pc))
    # mission looks: (time in h after peak_date 00 UTC, clear?) for every day offset
    iy=int((r.lat-24.5)/0.1); ix=int((r.lon+125)/0.1)
    pd0=dt.date.fromisoformat(r.peak_date)
    def k(o): return di.get((pd0+dt.timedelta(int(o))).isoformat())
    O=np.arange(-3,12)
    def looks(days,loc,cube=None):
        out=[]
        for o in days:
            tu=o*24+loc-r.lon/15
            if tu<tg[0] or tu>tg[-1]: continue
            out.append((tu, True if cube is None else clear(cube,k(o),iy,ix)))
        return out
    def best_of(L):
        ys=[yat(tu) for tu,c in L if c]; ys=[q for q in ys if np.isfinite(q)]
        return max(ys) if ys else 0.0
    def phased(rep,loc,cube=None,sar=False):  # list over phases of look lists
        res=[]
        for p in range(rep):
            days=[o for o in O if (o-p)%rep==0]
            if sar: res+= [looks(days,6), looks(days,18)]
            else: res.append(looks(days,loc,cube))
        return res
    modis=looks(O,10.5,T)+looks(O,13.5,A); modis_cf=looks(O,10.5)+looks(O,13.5)
    S2=phased(5,10.5,T); L89=phased(8,10.2,T); S1_12=phased(12,0,sar=True); S1_6=phased(6,0,sar=True); NI=phased(12,0,sar=True)
    SD=phased(1,0,sar=True)
    sc={}
    sc['MODIS Terra+Aqua']=[best_of(modis)]
    sc['MODIS, no cloud']=[best_of(modis_cf)]
    sc['Landsat 8/9 (8 d)']=[best_of(L) for L in L89]
    sc['Sentinel-2 (5 d)']=[best_of(L) for L in S2]
    sc['HLS (S2 + Landsat)']=[best_of(a_+b_) for a_,b_ in itertools.product(S2,L89)]
    sc['Sentinel-1 (12 d)']=[best_of(L) for L in S1_12]
    sc['Sentinel-1A/C (6 d)']=[best_of(L) for L in S1_6]
    sc['NISAR (12 d)']=[best_of(L) for L in NI]
    sc['S1 6 d + NISAR']=[best_of(a_+b_) for a_,b_ in itertools.product(S1_6,NI)]
    allc=[]
    for _ in range(200):
        allc.append(best_of(modis+S2[rng.integers(len(S2))]+L89[rng.integers(len(L89))]+S1_6[rng.integers(len(S1_6))]+NI[rng.integers(len(NI))]))
    sc['All current']=allc
    sc['Daily SAR']=[best_of(L) for L in SD]
    for nm,b in sc.items():
        b=np.array(b,dtype=float)
        rows.append(dict(site_no=r.site_no,peak_date=r.peak_date,A=r.A,scheme=nm,ext=ext(b).mean(),p80=(ext(b)>=0.8).mean(),pop=pop(b).mean(),pc=pc,yb=b.mean()))
    if i%50==0: print(i,flush=True)
S=pd.DataFrame(rows); W=pd.DataFrame(sweep); Du=pd.DataFrame(dur)
for X in (S,W,Du): X['ab']=pd.cut(X.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
S.to_parquet('data/osse_schemes.parquet'); W.to_parquet('data/osse_sweep.parquet'); Du.to_parquet('data/osse_duration.parquet')
tab=S.groupby(['scheme','ab'],observed=True).ext.mean().unstack();
pw=S.dropna(subset=['pc']).groupby('scheme').apply(lambda d:d['pop'].sum()/d.pc.sum()); tab['people']=pw; tab['all']=S.groupby('scheme').ext.mean()
print(tab.round(2).to_string()); tab.to_csv('fig/osse_schemes.csv')
print(W.groupby(['rv','ab'],observed=True).ext.mean().unstack().round(2).to_string())
print(Du.groupby('ab',observed=True)[['D50','D80','D90']].median().round(1).to_string())
