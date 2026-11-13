# Crest reconstruction: predict crest water depth from the depth at the first clear look,
# the lag in days and basin area (learned recession prior), leave-sites-out CV; compare with the naive "what the satellite saw"
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
R=pd.read_parquet('data/stage_first.parquet'); s=pd.read_parquet('data/main_sample.parquet')
s=s[['site_no','peak_date','first_clear','rise_days','fall_half_days','inst_to_daily','lat','lon','peak_date']].loc[:,lambda d:~d.columns.duplicated()]
M=R.merge(s,on=['site_no','peak_date']); M=M[(M.dh_peak_ft>0.3)&(M.dh_first_ft>0.05)&M.first_clear.notna()].copy()
M['mon']=M.peak_date.str[5:7].astype(int); M['lA']=np.log10(M.A); M['y']=np.log(M.dh_peak_ft/M.dh_first_ft)
M['lag']=M.first_clear
X1=M[['lag','lA']].assign(lagA=M.lag*M.lA).values
X2=M[['lag','lA','mon','lat','lon']].values
gkf=GroupKFold(5); p1=np.zeros(len(M)); p2=np.zeros(len(M))
for tr,te in gkf.split(X1,M.y,M.site_no):
    p1[te]=LinearRegression().fit(X1[tr],M.y.values[tr]).predict(X1[te])
    p2[te]=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,random_state=0).fit(X2[tr],M.y.values[tr]).predict(X2[te])
M['dpk_naive']=M.dh_first_ft; M['dpk_lin']=M.dh_first_ft*np.exp(p1); M['dpk_gb']=M.dh_first_ft*np.exp(p2)
def err(c): return (M[c]/M.dh_peak_ft)
M['abin2']=pd.cut(M.A,[0,30,100,300,1000,3000,10000,1e7],labels=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k'])
g=M.groupby('abin2',observed=True)
o=pd.DataFrame({'n':g.size(),'naive_med':g.apply(lambda d:(d.dpk_naive/d.dh_peak_ft).median()),'lin_med':g.apply(lambda d:(d.dpk_lin/d.dh_peak_ft).median()),'gb_med':g.apply(lambda d:(d.dpk_gb/d.dh_peak_ft).median()),
 'naive_within25':g.apply(lambda d:(abs(d.dpk_naive/d.dh_peak_ft-1)<=0.25).mean()),'gb_within25':g.apply(lambda d:(abs(d.dpk_gb/d.dh_peak_ft-1)<=0.25).mean()),
 'naive_mae_m':g.apply(lambda d:(abs(d.dpk_naive-d.dh_peak_ft)).median()*0.3048),'gb_mae_m':g.apply(lambda d:(abs(d.dpk_gb-d.dh_peak_ft)).median()*0.3048)})
print(o.round(2).to_string()); M.to_parquet('data/recon_cv.parquet')
