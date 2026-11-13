# OSSE checks: D50 truncation, event-split test of the duration/revisit curve, bootstrap intervals for schemes and required revisits
import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
S=pd.read_parquet('data/osse_schemes.parquet'); W=pd.read_parquet('data/osse_sweep.parquet'); Du=pd.read_parquet('data/osse_duration.parquet')
for X in (S,W,Du): X['ab']=X.ab.astype(str)
print('D50 median by class',Du.groupby('ab').D50.median().reindex(lab).round(1).to_dict())
print('D50>=250 h (window-limited)',(Du.D50>=250).sum(),'of',len(Du)); print('D50>144h by class',Du.groupby('ab').D50.apply(lambda x:(x>144).mean()).reindex(lab).round(2).to_dict())
print('D50<21h by class',Du.groupby('ab').D50.apply(lambda x:(x<21).mean()).reindex(lab).round(2).to_dict())
X=W.merge(Du[['site_no','peak_date','D50']],on=['site_no','peak_date']); X=X[X.D50>0].copy(); X['lx']=np.log10(X.D50/X.rv)
ev=X[['site_no','peak_date']].drop_duplicates().reset_index(drop=True); ev['id']=np.arange(len(ev)); X=X.merge(ev,on=['site_no','peak_date'])
bins=np.linspace(-2.5,2.5,26); rng=np.random.default_rng(0); R2=[]; R2a=[]
def curve_fit(d):
    d=d.assign(b=pd.cut(d.lx,bins)); g=d.groupby('b',observed=False).ext.mean(); c=np.array([iv.mid for iv in g.index]); v=g.values; m=np.isfinite(v); return c[m],v[m]
for it in range(200):
    tr=rng.random(len(ev))<0.5; a=X[X.id.isin(ev.id[tr])]; b=X[X.id.isin(ev.id[~tr])]
    c,v=curve_fit(a); pr=np.interp(b.lx,c,v); R2.append(1-((b.ext-pr)**2).sum()/((b.ext-b.ext.mean())**2).sum())
    # area-class-mean curves for comparison of the collapse: per-crest prediction from the event-split area-class curve
    pa=np.zeros(len(b))
    for cl in lab:
        aa=a[a.ab==cl]; gm=aa.groupby('rv').ext.mean(); k=(b.ab==cl).values; pa[k]=gm.reindex(b.rv[k]).values
    R2a.append(1-((b.ext-pa)**2).sum()/((b.ext-b.ext.mean())**2).sum())
print('event-split R2 duration/revisit curve: median',np.median(R2).round(3),'2.5-97.5',np.percentile(R2,[2.5,97.5]).round(3))
print('event-split R2 area-class x revisit means:',np.median(R2a).round(3))
# in-sample check (previous)
c,v=curve_fit(X); pr=np.interp(X.lx,c,v); print('in-sample R2',round(1-((X.ext-pr)**2).sum()/((X.ext-X.ext.mean())**2).sum(),3))
# residual by crest shape: rise vs recession asymmetry from D80 and 15-min not available here; use D90/D50 ratio as peakedness
X=X.merge(Du[['site_no','peak_date','D90']],on=['site_no','peak_date']); X['res']=X.ext-np.interp(X.lx,c,v)
X['peaked']=pd.qcut(X.D90/X.D50,3,labels=['peaked','mid','flat'])
print('mean residual by peakedness',X.groupby('peaked',observed=True).res.mean().round(3).to_dict())
# bootstrap by event for scheme people shares and class means
NB=1000; sch=S.scheme.unique(); keys=S[['site_no','peak_date']].drop_duplicates().reset_index(drop=True); keys['k']=np.arange(len(keys)); S=S.merge(keys,on=['site_no','peak_date'])
P=S.dropna(subset=['pc']); bs={s:[] for s in sch}
Pp=P.pivot_table(index='k',columns='scheme',values='pop'); Pc=P.groupby('k').pc.first()
for _ in range(NB):
    kk=rng.integers(0,len(keys),len(keys)); kk=kk[np.isin(kk,Pp.index)]
    tot=Pc.reindex(kk).sum()
    for s in sch: bs[s].append(Pp[s].reindex(kk).sum()/tot)
out=pd.DataFrame({s:{'people':P[P.scheme==s]['pop'].sum()/P[P.scheme==s].pc.sum(),'lo':np.percentile(bs[s],2.5),'hi':np.percentile(bs[s],97.5)} for s in sch}).T
print(out.round(3).to_string()); out.to_csv('fig/rev/osse_people_ci.csv')
# required revisit for 80% capture with bootstrap
def req(Wd):
    r={}
    G=Wd.groupby(['ab','rv']).ext.mean().unstack(0)
    for b in lab:
        vv=G[b].values; rv=G.index.values; k=np.where(vv<0.8)[0]
        r[b]=np.exp(np.interp(0.8,[vv[k[0]],vv[k[0]-1]],[np.log(rv[k[0]]),np.log(rv[k[0]-1])])) if len(k) and k[0]>0 else np.nan
    return r
r0=req(W); Wk=W.merge(keys,on=['site_no','peak_date']); bsr={b:[] for b in lab}
grp={k:d for k,d in Wk.groupby('k')}
for _ in range(300):
    kk=rng.integers(0,len(keys),len(keys)); Wd=pd.concat([grp[k] for k in kk if k in grp]); rr=req(Wd)
    for b in lab: bsr[b].append(rr[b])
RQ=pd.DataFrame({b:{'req_h':r0[b],'lo':np.nanpercentile(bsr[b],2.5),'hi':np.nanpercentile(bsr[b],97.5)} for b in lab}).T; print(RQ.round(1).to_string()); RQ.to_csv('fig/rev/osse_required_revisit_ci.csv')
# class means of extent captured with CI for MODIS actual and all current
for s in ['MODIS Terra+Aqua','All current','Daily SAR']:
    d=S[S.scheme==s]; print(s,{b:round(d[d.ab==b].ext.mean(),2) for b in lab})
