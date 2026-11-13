# Additional checks: S1 subset, missing pairs, gauge dedupe, strict threshold, GOES timing offsets, global rain window, sample counts, MODIS threshold cross-fit
import numpy as np, pandas as pd
pd.set_option('display.width',250)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; bins=[0,30,100,300,1000,3000,10000,1e7]
s=pd.read_parquet('data/main_sample.parquet'); s['ab']=pd.cut(s.drain_area_km2,bins,labels=lab)
U=pd.read_parquet('data/global/usgs_goes_modis.parquet'); U['ab']=U.ab.astype(str)
G1=pd.read_parquet('data/goes1deg_events.parquet')
print('main',len(s),'U',len(U))
# T1: 1-degree GOES on the matched subset
M=U.merge(G1[['site_no','peak_date','c0','clim1']],on=['site_no','peak_date'],how='left')
rg=lambda d:d.b0.sum()/d.clim_b.sum()
out=[]
for b in lab:
    d=M[M.ab==b]; dd=d.dropna(subset=['c0','clim1','b0'])
    rm=0.5*(d.T_0.mean()/d.clim_T.mean()+d.A_0.mean()/d.clim_A.mean())
    out.append([b,len(d),rg(d),dd.b0.sum()/dd.clim_b.sum(),dd.c0.sum()/dd.clim1.sum(),rm,len(dd)])
T1=pd.DataFrame(out,columns=['ab','n','goes01','goes01_sub1','goes1','modis','n1']); print(T1.round(3)); T1.to_csv('fig/rev/S1_subset.csv',index=False)
# T2: why 14662 vs 14670
g=U.b0.values; m=np.fmax(U.T_0.values,U.A_0.values)
print('missing goes b0',np.isnan(g).sum(),'missing modis both',np.isnan(m).sum(),'either',(np.isnan(g)|np.isnan(m)).sum())
# T3: dedupe gauges
c=pd.read_parquet('data/global/caravan_ana.parquet')
print(c.dataset.value_counts())
sites=c.groupby(['dataset','basin_id']).agg(lat=('lat','first'),lon=('lon','first')).reset_index()
us=s.groupby('site_no').agg(lat=('lat','first'),lon=('lon','first')).reset_index()
def idnum(x):
    import re; m=re.search(r'(\d{8,15})$',x); return m.group(1) if m else None
sites['num']=sites.basin_id.map(idnum)
print(sites[sites.dataset.str.contains('camels$|hysets',regex=True)].head())
ov_id=sites.num.isin(set(us.site_no))
# coordinate match within 0.005 deg
from scipy.spatial import cKDTree
t=cKDTree(us[['lat','lon']].values); dist,_=t.query(sites[['lat','lon']].values)
ov_xy=dist<0.005
print('caravan sites',len(sites),'usgs main sites',len(us),'overlap by id',ov_id.sum(),'by xy',ov_xy.sum(),'either',(ov_id|ov_xy).sum())
print(sites[ov_id|ov_xy].dataset.value_counts())
# duplicates within caravan (camels-us vs hysets)
t2=cKDTree(sites[['lat','lon']].values); pairs=t2.query_pairs(0.005); print('within-caravan near-duplicate pairs',len(pairs))
dup=set()
for i,j in pairs: dup.add(max(i,j))
print('unique gauges total', len(us)+len(sites)-(ov_id|ov_xy).sum()-len([k for k in dup if not (ov_id|ov_xy)[k]]))
# T4: strict clear threshold
T=np.load('data/terra.npy',mmap_mode='r'); A=np.load('data/aqua.npy',mmap_mode='r'); D=np.load('data/days.npy',allow_pickle=True)
mon=np.array([int(str(d)[5:7]) for d in D])
iy=s.iy.values; ix=s.ix.values
cells=np.unique(np.stack([iy,ix],1),axis=0)
clim90=np.full((12,len(cells)),np.nan); clim50=clim90.copy()
tv=T[:,cells[:,0],cells[:,1]].astype('f4'); av=A[:,cells[:,0],cells[:,1]].astype('f4')
tv[tv==255]=np.nan; av[av==255]=np.nan; bv=np.fmax(tv,av)/200
for mm in range(1,13):
    x=bv[mon==mm]; v=np.isfinite(x)
    clim90[mm-1]=np.where(v,x>=0.9,0).sum(0)/v.sum(0); clim50[mm-1]=np.where(v,x>=0.5,0).sum(0)/v.sum(0)
ci={tuple(k):i for i,k in enumerate(cells)}; k=np.array([ci[(a,b)] for a,b in zip(iy,ix)]); mm=np.array([int(d[5:7]) for d in s.peak_date])-1
s['p90']=clim90[mm,k]; s['p50']=clim50[mm,k]
out=[]
for b in lab:
    d=s[s.ab==b]; v=d.b0.notna()
    out.append([b,(d.b0[v]>=0.9).sum()/d.p90[v].sum(),(d.b0[v]>=0.5).sum()/d.p50[v].sum(),d.b0.sum()/d.clim_b[v].sum()])
T4=pd.DataFrame(out,columns=['ab','r90','r50','r_frac']); print(T4.round(3)); T4.to_csv('fig/rev/strict_threshold.csv',index=False)
# first clear look with 0.9
B=s[[f'b{o}' for o in range(0,11)]].values; Q=s[[f'q{o}' for o in range(0,11)]].values
def first(th):
    hit=np.nan_to_num(B,nan=-1)>=th; kk=np.where(hit.any(1),hit.argmax(1),-1); return np.where(kk>=0,Q[np.arange(len(s)),np.clip(kk,0,None)],np.nan), np.where(kk>=0,kk,np.nan)
for th in [0.3,0.5,0.7,0.9]:
    q,lag=first(th); print(th,[round(np.nanmedian(q[(s.ab==b).values]),2) for b in lab],'never',round(np.isnan(q).mean(),3))
# T5: GOES offset of the sampled scan from 10:30/13:30 local solar time
lonc=-125+0.1*(s.ix.values+0.5)
for loc in [10.5,13.5]:
    utc=loc-lonc/15; h=np.clip(np.round(utc-0.52),15,21); scan=h+33.5/60  # scan mid-time
    off=(scan-utc)*60; s[f'off{int(loc)}']=off
    print('offset min',loc,np.percentile(off,[0,5,25,50,75,95,100]).round(1),'abs median',np.median(abs(off)).round(1))
s['absoff']=np.maximum(abs(s.off10),abs(s.off13))
for nm,sel in [('|off|<15',abs(s.off10)<15),('15-30',abs(s.off10)>=15)]:
    d=s[sel]; print(nm,len(d),[round(d[d.ab==b].t0.sum()/d[d.ab==b].clim_t.sum(),2) for b in lab])
# T6: global rain window excluding post-crest rain
c['ab']=pd.cut(c.area_km2,bins,labels=lab)
rr=lambda d:0.5*(d.T_0.mean()/d.clim_T.mean()+d.A_0.mean()/d.clim_A.mean())
c['P_pre']=c[['P_m3','P_m2','P_m1','P_0']].max(1); c['P_all']=c[['P_m3','P_m2','P_m1','P_0','P_p1']].max(1)
out=[]
for b in lab:
    d=c[c.ab==b]; out.append([b,(d.P_all>=10).sum(),rr(d[d.P_all>=10]),(d.P_pre>=10).sum(),rr(d[d.P_pre>=10]),rr(d[(d.P_pre>=10)&(d.P_p1<1)]),(d.P_all<1).sum(),rr(d[d.P_all<1])])
T6=pd.DataFrame(out,columns=['ab','n_storm','r_storm','n_pre','r_pre','r_pre_dry_after','n_norain','r_norain']); print(T6.round(3)); T6.to_csv('fig/rev/global_window.csv',index=False)
print('rain col check rain==', c.rain.value_counts().to_dict() if 'rain' in c else '')
# Counts per dataset x area class
N=c.groupby(['dataset','ab'],observed=False).agg(n=('basin_id','size'),g=('basin_id','nunique')).unstack('ab'); print(N)
N.to_csv('fig/rev/counts_dataset_area.csv')
nu=s.groupby('ab',observed=False).agg(n=('site_no','size'),g=('site_no','nunique')); print(nu); nu.to_csv('fig/rev/counts_usgs_area.csv')
# T8: MODIS threshold cross-fit (east vs west, odd vs even years) and confusion matrix
O=range(0,11); QU=np.stack([U[f'q{o}'].values for o in O],1)
key=lambda o:'0' if o==0 else (f'm{-o}' if o<0 else f'p{o}')
BG=np.stack([U[f'b{o}'].values for o in O],1); BM=np.stack([np.fmax(U[f'T_{key(o)}'].values,U[f'A_{key(o)}'].values) for o in O],1)
def firstU(Bx,th,sel):
    Bs=Bx[sel]; hit=np.nan_to_num(Bs,nan=-1)>=th; kk=np.where(hit.any(1),hit.argmax(1),-1); qq=np.where(kk>=0,QU[sel][np.arange(sel.sum()),np.clip(kk,0,None)],np.nan); return qq,np.where(kk>=0,kk,np.nan)
ths=np.round(np.arange(0.2,0.71,0.05),2)
def fitth(sel):
    qg,_=firstU(BG,0.5,sel); best=None
    for th in ths:
        qm,_=firstU(BM,th,sel); err=np.nanmean([abs(np.nanmedian(qm[(U.ab[sel]==b).values])-np.nanmedian(qg[(U.ab[sel]==b).values])) for b in lab])
        if best is None or err<best[0]: best=(err,th)
    return best[1]
yr=U.peak_date.str[:4].astype(int).values; east=(U.lon>-100).values
for nm,sa,sb in [('east->west',east,~east),('west->east',~east,east),('odd->even',yr%2==1,yr%2==0),('even->odd',yr%2==0,yr%2==1)]:
    th=fitth(sa); qg,lg=firstU(BG,0.5,sb); qm,lm=firstU(BM,th,sb)
    print(nm,'th',th,'test |diff| by class',[round(np.nanmedian(qm[(U.ab[sb]==b).values])-np.nanmedian(qg[(U.ab[sb]==b).values]),3) for b in lab])
# confusion matrix day-level clear classification and wait-time difference
gg=(BG>=0.5); mmx=(BM>=0.4); v=np.isfinite(BG)&np.isfinite(BM)
tp=(gg&mmx&v).sum(); tn=(~gg&~mmx&v).sum(); fp=(~gg&mmx&v).sum(); fn=(gg&~mmx&v).sum()
print('confusion GOES>=0.5 vs MODIS>=0.4: agree',round((tp+tn)/v.sum(),3),'tp tn fp fn',tp,tn,fp,fn)
all_=np.ones(len(U),bool); _,lg=firstU(BG,0.5,all_); _,lm=firstU(BM,0.4,all_); dl=lm-lg; ok=np.isfinite(dl)
print('wait diff days: same',round((dl[ok]==0).mean(),3),'within 1 day',round((abs(dl[ok])<=1).mean(),3),'median',np.median(dl[ok]))
for b in lab:
    sel=(U.ab==b).values&ok; print(b,'same',round((dl[sel]==0).mean(),2),'|d|<=1',round((abs(dl[sel])<=1).mean(),2))
s[['site_no','peak_date','p90','p50','off10','off13']].to_parquet('data/rev_quick_extra.parquet')
