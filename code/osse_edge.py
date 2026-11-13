# Crest-duration lower bounds: flag crests whose half-extent run touches the start or end of the 15-minute window,
# and repeat the duration/revisit curve, its held-out R2 and the revisit needed for 80% capture without them.
import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; BINS=[0,30,100,300,1000,3000,10000,1e7]
v=pd.read_parquet('data/usgs/iv.parquet'); v=v[v.param=='00065'].dropna(subset=['value'])
E=pd.read_parquet('data/iv_capture_ext.parquet'); E=E[np.isfinite(E.ext_cloudfree_max)].reset_index(drop=True)
DH=np.array([0,0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6,8,10,12,15]); hc=[f'h{d}' for d in DH]
G={k:g.sort_values('time_utc') for k,g in v.groupby(['site_no','peak_date'])}; rows=[]
for _,r in E.iterrows():
    g=G[(r.site_no,r.peak_date)]; t=(g.time_utc-pd.Timestamp(r.peak_date,tz='UTC')).dt.total_seconds().values/3600; h=g.value.values
    ic=np.argmax(h); tc=t[ic]; pre=(t<tc)&(t>=tc-72); hb=np.min(h[pre]) if pre.sum()>4 else np.min(h[:ic+1])
    tg=np.arange(max(t[0],tc-48),min(t[-1],tc+240),0.25); y=np.clip((np.interp(tg,t,h)-hb)/(h[ic]-hb),0,1)
    a=r[hc].values.astype(float); full=np.interp(r.depth_m,DH,a)-a[0]; ey=(np.interp(y*r.depth_m,DH,a)-a[0])/full
    k0=np.argmin(np.abs(tg-tc)); lo=k0; hi=k0
    while lo>0 and ey[lo-1]>=0.5: lo-=1
    while hi<len(ey)-1 and ey[hi+1]>=0.5: hi+=1
    rows.append(dict(site_no=r.site_no,peak_date=r.peak_date,edge_start=lo==0,edge_end=hi==len(ey)-1))
F=pd.DataFrame(rows); F['edge']=F.edge_start|F.edge_end; F.to_parquet('data/osse_edge.parquet')
print('edge start',F.edge_start.sum(),'end',F.edge_end.sum(),'both',(F.edge_start&F.edge_end).sum(),'any',F.edge.sum(),'of',len(F))
W=pd.read_parquet('data/osse_sweep.parquet'); Du=pd.read_parquet('data/osse_duration.parquet')
for X in (W,Du): X['ab']=X.ab.astype(str)
bins=np.linspace(-2.5,2.5,26)
def curve_fit(d):
    d=d.assign(b=pd.cut(d.lx,bins)); g=d.groupby('b',observed=False).ext.mean(); c=np.array([iv.mid for iv in g.index]); v_=g.values; m=np.isfinite(v_); return c[m],v_[m]
def req(Wd):
    r={}; Gm=Wd.groupby(['ab','rv']).ext.mean().unstack(0)
    for b in lab:
        vv=Gm[b].values; rv=Gm.index.values; k=np.where(vv<0.8)[0]
        r[b]=np.exp(np.interp(0.8,[vv[k[0]],vv[k[0]-1]],[np.log(rv[k[0]]),np.log(rv[k[0]-1])])) if len(k) and k[0]>0 else np.nan
    return r
def run(keep,name,nb=300):
    X=W.merge(Du[['site_no','peak_date','D50']],on=['site_no','peak_date']).merge(keep,on=['site_no','peak_date']); X=X[X.D50>0].copy(); X['lx']=np.log10(X.D50/X.rv)
    ev=X[['site_no','peak_date']].drop_duplicates().reset_index(drop=True); ev['id']=np.arange(len(ev)); X=X.merge(ev,on=['site_no','peak_date'])
    rng=np.random.default_rng(0); R2=[]; R2a=[]
    for it in range(200):
        tr=rng.random(len(ev))<0.5; a=X[X.id.isin(ev.id[tr])]; b=X[X.id.isin(ev.id[~tr])]
        c,v_=curve_fit(a); pr=np.interp(b.lx,c,v_); R2.append(1-((b.ext-pr)**2).sum()/((b.ext-b.ext.mean())**2).sum())
        pa=np.zeros(len(b))
        for cl in lab:
            aa=a[a.ab==cl]; gm=aa.groupby('rv').ext.mean(); k=(b.ab==cl).values; pa[k]=gm.reindex(b.rv[k]).values
        R2a.append(1-((b.ext-pa)**2).sum()/((b.ext-b.ext.mean())**2).sum())
    Wd=W.merge(keep,on=['site_no','peak_date']); r0=req(Wd); keys=Wd[['site_no','peak_date']].drop_duplicates().reset_index(drop=True); keys['k']=np.arange(len(keys))
    grp={k:d for k,d in Wd.merge(keys,on=['site_no','peak_date']).groupby('k')}; bsr={b:[] for b in lab}
    for _ in range(nb):
        kk=rng.integers(0,len(keys),len(keys)); rr=req(pd.concat([grp[k] for k in kk]))
        for b in lab: bsr[b].append(rr[b])
    n=Du.merge(keep,on=['site_no','peak_date']).groupby('ab').size().reindex(lab)
    D=Du.merge(keep,on=['site_no','peak_date']); fit=np.polyfit(np.log(D.A),np.log(D.D50),1)[0]; rr_=np.corrcoef(np.log(D.A),np.log(D.D50))[0,1]
    out=dict(set=name,crests=len(ev),R2_curve=np.median(R2),R2_lo=np.percentile(R2,2.5),R2_hi=np.percentile(R2,97.5),R2_area=np.median(R2a),D50_exp=fit,D50_r=rr_)
    for b in lab: out[f'n_{b}']=n[b]; out[f'req_{b}']=r0[b]; out[f'lo_{b}']=np.nanpercentile(bsr[b],2.5); out[f'hi_{b}']=np.nanpercentile(bsr[b],97.5)
    for b in lab: out[f'D50_{b}']=D[D.ab==b].D50.median()
    return out
allk=F[['site_no','peak_date']]; nok=F.loc[~F.edge,['site_no','peak_date']]
T=pd.DataFrame([run(allk,'all 390'),run(nok,'without 41 edge-limited')]).set_index('set').T
pd.set_option('display.width',200); print(T.round(3).to_string()); T.to_csv('fig/rev/osse_edge_sensitivity.csv')
