# Global test with Caravan annual-maximum crests and MODIS daily 1-degree clear-sky fraction
# usage: python global_ana.py <dir with matched parquet files>
import numpy as np, pandas as pd, glob, sys, re
D=sys.argv[1] if len(sys.argv)>1 else 'data/global'
fs=sorted(glob.glob(f'{D}/*.parquet')); print(fs)
T=pd.concat([pd.read_parquet(f).assign(src=f.split('/')[-1]) for f in fs],ignore_index=True)
print(T.shape); print(list(T.columns)[:60])
def off(c):  # T_m3 -> -3, T_p10 -> 10, T_0 -> 0
    m=re.match(r'^([TAqP])_(m|p)?(\d+)$',c); 
    return None if not m else (m.group(1),(-1 if m.group(2)=='m' else 1)*int(m.group(3)))
OFF=sorted({off(c)[1] for c in T.columns if off(c) and off(c)[0]=='T'})
area=[c for c in T.columns if c.lower() in ('area_km2','area','drain_area_km2')][0]
ds=[c for c in T.columns if c.lower() in ('dataset','source','ds','collection')]; ds=ds[0] if ds else 'src'
def col(k,o): 
    for c in (f'{k}_{"m" if o<0 else "p"}{abs(o)}',f'{k}_{o}',f'{k}_p0',f'{k}_m0'):
        if c in T.columns: return c
for o in OFF: T[f'r{o}']=0.5*(T[col('T',o)]/T.clim_T+T[col('A',o)]/T.clim_A)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; T['ab']=pd.cut(T[area],[0,30,100,300,1000,3000,10000,1e7],labels=lab)
def ratio(d,o): return 0.5*(d[col('T',o)].mean()/d.clim_T.mean()+d[col('A',o)].mean()/d.clim_A.mean())
print('\ncrest-day clear/normal by area and dataset')
tab=T.groupby([ds,'ab'],observed=True).apply(lambda d:pd.Series({'n':len(d),'r0':ratio(d,0)})).unstack('ab')
print(tab.round(2).to_string())
