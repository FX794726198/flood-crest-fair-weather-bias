import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
pd.set_option('display.width',250)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; BINS=[0,30,100,300,1000,3000,10000,1e7]
D=pd.read_parquet('data/rev_depth_days.parquet'); RC=pd.read_parquet('data/recon_cv.parquet')
s=pd.read_parquet('data/main_sample.parquet')[['site_no','peak_date']+[f'b{o}' for o in range(-3,11)]]
D=D.merge(s,on=['site_no','peak_date'])
FT=0.3048
D['ab']=pd.cut(D.A,BINS,labels=lab).astype(str); D['lA']=np.log10(D.A); D['mon']=D.peak_date.str[5:7].astype(int)
D['season']=D.mon.map(lambda m:(m%12)//3)
fc=D.first_clear.values
D['d1']=[r[f'd{int(k)}'] if np.isfinite(k) else np.nan for k,(_,r) in zip(fc,D.iterrows())]
D['d1L']=[r[f'dL{int(k)}'] if np.isfinite(k) else np.nan for k,(_,r) in zip(fc,D.iterrows())]
excl={'rated crests':len(D),'no clear look within 10 days':int(D.first_clear.isna().sum())}
ok=(D.dpk>0.3)&(D.d1>0.05)&D.first_clear.notna(); excl['crest depth <=0.3 ft']=int((D.dpk<=0.3).sum()); excl['first-look depth <=0.05 ft (zero/negative)']=int(((D.d1<=0.05)&D.first_clear.notna()&(D.dpk>0.3)).sum())
M=D[ok].reset_index(drop=True); excl['kept']=len(M); print(excl)
M['y']=np.log(M.dpk/M.d1); M['lag']=M.first_clear
gkf=list(GroupKFold(5).split(M,M.y,M.site_no))
def gb(cols,target,data=M,q=None):
    p=np.zeros(len(data)); X=data[cols].values; y=data[target].values
    for tr,te in GroupKFold(5).split(X,y,data.site_no):
        m=HGB(loss='quantile',quantile=q,max_iter=300,learning_rate=0.05,random_state=0) if q else HGB(max_iter=300,learning_rate=0.05,random_state=0)
        p[te]=m.fit(X[tr],y[tr]).predict(X[te])
    return p
base=['lag','lA','mon','lat','lon']
M['est_gb']=M.d1*np.exp(gb(base,'y'))
# baselines fitted within training folds
b1=np.zeros(len(M)); b2=b1.copy()
for tr,te in gkf:
    T=M.iloc[tr]; t1=T.groupby(['ab','season']).y.median(); t2=T.groupby(T.lag.clip(0,10)).y.median()
    b1[te]=t1.reindex(list(zip(M.ab.values[te],M.season.values[te]))).fillna(T.y.median()).values
    b2[te]=t2.reindex(M.lag.clip(0,10).values[te]).fillna(T.y.median()).values
M['est_areamonth']=M.d1*np.exp(b1); M['est_wait']=M.d1*np.exp(b2)
M['ldpk']=np.log(M.dpk); M['est_prior_only']=np.exp(gb(['lA','mon','lat','lon'],'ldpk'))
# timing error: model trained on true lag, applied with lag shifted by +-1 day
pe=np.zeros(len(M)); pm=pe.copy()
X=M[base].values
for tr,te in gkf:
    m=HGB(max_iter=300,learning_rate=0.05,random_state=0).fit(X[tr],M.y.values[tr])
    Xp=X[te].copy(); Xp[:,0]+=1; pe[te]=m.predict(Xp); Xm=X[te].copy(); Xm[:,0]=np.maximum(Xm[:,0]-1,0); pm[te]=m.predict(Xm)
M['est_lag+1']=M.d1*np.exp(pe); M['est_lag-1']=M.d1*np.exp(pm)
# quantile intervals
for qq in (0.1,0.9,0.05,0.95): M[f'q{int(qq*100)}']=M.d1*np.exp(gb(base,'y',q=qq))
M['cov80']=(M.dpk>=M.q10)&(M.dpk<=M.q90); M['cov90']=(M.dpk>=M.q5)&(M.dpk<=M.q95)
# nested cross-validated residual-quantile intervals: residual quantiles by area class from inner grouped folds of the training gauges
lo80=np.zeros(len(M)); hi80=lo80.copy(); lo90=lo80.copy(); hi90=lo80.copy(); Xb=M[base].values; yb=M.y.values
for tr,te in gkf:
    res=np.full(len(tr),np.nan); sites=M.site_no.values[tr]
    for itr,ite in GroupKFold(5).split(Xb[tr],yb[tr],sites):
        res[ite]=yb[tr][ite]-HGB(max_iter=300,learning_rate=0.05,random_state=0).fit(Xb[tr][itr],yb[tr][itr]).predict(Xb[tr][ite])
    m=HGB(max_iter=300,learning_rate=0.05,random_state=0).fit(Xb[tr],yb[tr]); pt=m.predict(Xb[te])
    abt=M.ab.values[tr]; abe=M.ab.values[te]
    for b in set(abe):
        rb=res[abt==b]; k=abe==b
        q=np.quantile(rb,[0.1,0.9,0.05,0.95]); lo80[te[k]]=pt[k]+q[0]; hi80[te[k]]=pt[k]+q[1]; lo90[te[k]]=pt[k]+q[2]; hi90[te[k]]=pt[k]+q[3]
M['cf_cov80']=(M.y>=lo80)&(M.y<=hi80); M['cf_cov90']=(M.y>=lo90)&(M.y<=hi90); M['cf_w80']=np.exp(hi80-lo80)
print('interval coverage 80%',M.cf_cov80.mean().round(3),'90%',M.cf_cov90.mean().round(3)); print(M.groupby('ab').agg(c80=('cf_cov80','mean'),c90=('cf_cov90','mean'),w80=('cf_w80','median')).reindex(lab).round(2).T)
# leave-own-peak-out rating curves
L=M[np.isfinite(M.dpk_L)&np.isfinite(M.d1L)&(M.dpk_L>0.3)&(M.d1L>0.05)].reset_index(drop=True); L['yL']=np.log(L.dpk_L/L.d1L)
L['est_L']=L.d1L*np.exp(gb(base,'yL',data=L))
def summ(d,est,truth='dpk'):
    rr=d[est]/d[truth]; return pd.Series({'n':len(d),'med_ratio':rr.median(),'w25':(abs(rr-1)<=0.25).mean(),'mae_m':abs(d[est]-d[truth]).median()*FT,'p10':rr.quantile(.1),'p90':rr.quantile(.9),'over25':(rr>1.25).mean(),'under25':(rr<0.75).mean()})
ests=['d1','est_areamonth','est_wait','est_prior_only','est_gb','est_lag+1','est_lag-1']
tab=pd.DataFrame({e:summ(M,e) for e in ests}).T; print(tab.round(3).to_string()); tab.to_csv('fig/rev/recon_baselines_all.csv')
rows=[]
for b in lab:
    d=M[M.ab==b]
    for e in ests: rows.append(dict(ab=b,est=e,**summ(d,e).to_dict()))
TB=pd.DataFrame(rows); print(TB.pivot(index='ab',columns='est',values='w25').reindex(lab).round(2).to_string()); print(TB.pivot(index='ab',columns='est',values='mae_m').reindex(lab).round(2).to_string())
TB.to_csv('fig/rev/recon_baselines_by_area.csv',index=False)
print('coverage 80%',M.cov80.mean().round(3),'90%',M.cov90.mean().round(3)); print(M.groupby('ab').agg(c80=('cov80','mean'),c90=('cov90','mean')).reindex(lab).round(2).T)
print('LOYO: full',summ(L,'d1L','dpk_L').round(3).to_dict()); print('LOYO rebuilt',summ(L,'est_L','dpk_L').round(3).to_dict())
print('crest depth LOYO vs full: median |diff| m',(abs(L.dpk_L-L.dpk)*FT).median().round(3),'p90',(abs(L.dpk_L-L.dpk)*FT).quantile(.9).round(3))
# error by lag
M['lagc']=M.lag.clip(0,4)
E=M.groupby(['ab','lagc']).apply(lambda d:pd.Series({'n':len(d),'w25_naive':(abs(d.d1/d.dpk-1)<=.25).mean(),'w25_gb':(abs(d.est_gb/d.dpk-1)<=.25).mean(),'mae_gb':abs(d.est_gb-d.dpk).median()*FT})).reset_index()
print(E.pivot(index='ab',columns='lagc',values='w25_gb').reindex(lab).round(2)); E.to_csv('fig/rev/recon_by_area_lag.csv',index=False)
M.to_parquet('data/rev_recon.parquet')
# ------ timing from radar storm day (no gauge timing): first clear look counted from the MRMS storm day ------
K=pd.read_parquet('data/mech.parquet')[['site_no','peak_date','kr']]; K['ks']=K.kr-1   # storm day relative to crest day
S=D.merge(K,on=['site_no','peak_date']); S=S[S.dpk>0.3].reset_index(drop=True)
Bm=S[[f'b{o}' for o in range(-3,11)]].values
lk=np.full(len(S),np.nan); dd=lk.copy()
for i in range(len(S)):
    for o in range(max(int(S.ks[i]),0),11):    # looks from the storm day (or crest day if storm day is earlier than the depth record) onward
        if np.isfinite(Bm[i,o+3]) and Bm[i,o+3]>=0.5 and np.isfinite(S[f'd{o}'][i]):
            lk[i]=o-S.ks[i]; dd[i]=S[f'd{o}'][i]; break
S['lag_storm']=lk; S['dlook']=dd
S=S[np.isfinite(S.dlook)&(S.dlook>0.05)].reset_index(drop=True); S['y']=np.log(S.dpk/S.dlook)
X=S[['lag_storm','lA','mon','lat','lon']].values; p=np.zeros(len(S))
for tr,te in GroupKFold(5).split(X,S.y,S.site_no): p[te]=HGB(max_iter=300,learning_rate=0.05,random_state=0).fit(X[tr],S.y.values[tr]).predict(X[te])
S['est_storm']=S.dlook*np.exp(p)
print('storm-timed: n',len(S)); print('naive',summ(S,'dlook').round(3).to_dict()); print('rebuilt',summ(S,'est_storm').round(3).to_dict())
rows=[]
for b in lab:
    d=S[S.ab==b]; rows.append(dict(ab=b,**{('naive_'+k):v for k,v in summ(d,'dlook').items()},**{('storm_'+k):v for k,v in summ(d,'est_storm').items()}))
ST=pd.DataFrame(rows); print(ST[['ab','naive_n','naive_w25','storm_w25','naive_mae_m','storm_mae_m','naive_med_ratio','storm_med_ratio']].round(2).to_string()); ST.to_csv('fig/rev/recon_stormtimed_by_area.csv',index=False)
S.to_parquet('data/rev_recon_storm.parquet')
