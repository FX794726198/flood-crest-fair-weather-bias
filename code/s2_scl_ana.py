import numpy as np, pandas as pd
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']
R=pd.read_parquet('data/s2/s2_scl.parquet'); print('rows',len(R),'errors',(R.day==-99).sum())
R=R[(R.day>=0)&(R.cov1>=0.9)&R.goes_t.notna()].copy(); print('valid scene looks',len(R),'crests',R[['site_no','peak_date']].drop_duplicates().shape[0])
g=R.goes_t>=0.5; s1=R.clr1>=0.5; s3=R.clr3>=0.5
acc=(g==s1).mean(); print('agreement GOES>=0.5 vs S2 1km clear>=0.5',round(acc,3),'kappa-ish', 'tp',(g&s1).sum(),'tn',(~g&~s1).sum(),'fp(GOES clear,S2 cloudy)',(g&~s1).sum(),'fn',(~g&s1).sum())
print('3 km agreement',round((g==s3).mean(),3)); print('r(goes,clr1)',round(np.corrcoef(R.goes_t,R.clr1)[0,1],3),'r(goes,clr3)',round(np.corrcoef(R.goes_t,R.clr3)[0,1],3))
print('P(S2 usable | GOES clear)',round(s1[g].mean(),3),'P(S2 usable | GOES cloudy)',round(s1[~g].mean(),3))
for th in [0.9]:
    print('strict S2 clear>=0.9: P|GOES clear',round((R.clr1[g]>=th).mean(),3),'P|GOES cloudy',round((R.clr1[~g]>=th).mean(),3))
R['ab']=R.ab.astype(str); R['early']=R.day<=1
T=R.groupby(['ab','early']).apply(lambda d:pd.Series({'n':len(d),'s2_clear':(d.clr1>=0.5).mean(),'goes_clear':(d.goes_t>=0.5).mean(),'agree':((d.goes_t>=0.5)==(d.clr1>=0.5)).mean()})).reset_index()
print(T.pivot(index='ab',columns='early',values='s2_clear').reindex(lab).round(2)); print(T.pivot(index='ab',columns='early',values='goes_clear').reindex(lab).round(2)); print(T.pivot(index='ab',columns='early',values='n').reindex(lab))
print(R.groupby('ab').apply(lambda d:((d.goes_t>=0.5)==(d.clr1>=0.5)).mean()).reindex(lab).round(3).to_dict())
T.to_csv('fig/rev/s2_scl_summary.csv',index=False)
# first usable S2 look after the crest: GOES-flagged clear vs S2 SCL-clear, same scene dates
F=R.sort_values('day').groupby(['site_no','peak_date']).apply(lambda d:pd.Series({'ab':d.ab.iloc[0],'first_s2':d.day[d.clr1>=0.5].min(),'first_goes':d.day[d.goes_t>=0.5].min()})).reset_index()
F=F.dropna(); print('first usable S2 date identical',round((F.first_s2==F.first_goes).mean(),3),'n',len(F))
