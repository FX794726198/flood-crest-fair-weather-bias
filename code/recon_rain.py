import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
RC=pd.read_parquet('data/recon_cv.parquet'); F=pd.read_parquet('data/mrms_feat.parquet')
M=RC.merge(F[['site_no','peak_date','P_mean48','P_max24','P_mean_d0','P_mean_d1']],on=['site_no','peak_date'])
M=M[np.isfinite(M.P_mean48)].copy(); M['lP']=np.log1p(M.P_mean48); M['lPx']=np.log1p(M.P_max24)
base=['lag','lA','mon','lat','lon']; rain=base+['lP','lPx','P_mean_d0','P_mean_d1']
out={}
for nm,cols in [('base',base),('rain',rain)]:
    X=M[cols].values; y=M.y.values; p=np.zeros(len(M))
    for tr,te in GroupKFold(5).split(X,y,M.site_no):
        p[te]=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0).fit(X[tr],y[tr]).predict(X[te])
    M['est_'+nm]=M.dh_first_ft*np.exp(p)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; M['ab']=pd.cut(M.A,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
g=M.groupby('ab',observed=True)
o=pd.DataFrame({'n':g.size()})
for c in ['dpk_naive','est_base','est_rain']:
    o[c+'_med']=g.apply(lambda d:(d[c]/d.dh_peak_ft).median()); o[c+'_w25']=g.apply(lambda d:(abs(d[c]/d.dh_peak_ft-1)<=.25).mean()); o[c+'_mae']=g.apply(lambda d:abs(d[c]-d.dh_peak_ft).median()*.3048)
print(o.round(2).to_string()); M.to_parquet('data/recon_rain.parquet')
