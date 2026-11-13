# Fig 3: two-timescale mechanism (basin crest lag vs post-storm cloud clearing)
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'axes.spines.top':False,'axes.spines.right':False})
rng=np.random.default_rng(1)
M=pd.read_parquet('data/mech.parquet')
M['L']=M.lag+1                      # crest day minus storm day; storm day = rain-file date - 1 (24-h QPE ends 12Z)
lab=['<30','30-100','100-300','300-1k','1k-3k','3k-10k','>10k']; labp=['<30','30–100','100–300','300–1k','1k–3k','3k–10k','>10k']
M['ab']=pd.cut(M.drain_area_km2,[0,30,100,300,1000,3000,10000,1e7],labels=lab)
M['season']=pd.to_datetime(M.peak_date).dt.month.map(lambda m:['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'][m-1])
M['reg']=np.where(M.lon<-100,'W','E')
JP=np.arange(-2,8)
# storm-aligned clear-sky matrix: E[e, jp] = b at offset kr+jp-1 from crest
B=np.stack([M[f'b{o}'].values for o in range(-3,11)],1)   # offsets -3..10
o=M.kr.values[:,None]+JP[None,:]-1; ok=(o>=-3)&(o<=10)
E=np.where(ok,np.take_along_axis(B,np.clip(o+3,0,13),1),np.nan); C=M.clim_b.values
def curve(idx):
    e=E[idx]; c=np.broadcast_to(C[idx,None],e.shape); m=np.isfinite(e)
    return np.where(m,e,0).sum(0)/np.where(m,c,0).sum(0)
sites=M.site_no.values; usite,inv=np.unique(sites,return_inverse=True); members=[np.where(inv==k)[0] for k in range(len(usite))]
def boot_idx(idx):
    s=np.unique(inv[idx]); ss=rng.choice(s,len(s)); return np.concatenate([np.intersect1d(members[k],idx,assume_unique=True) for k in ss])
def tclear(cv):                     # cloud-clearing time: first time after storm day the clear/normal ratio recovers to 0.75
    for k in range(np.where(JP==0)[0][0],len(JP)-1):
        if cv[k]<0.75<=cv[k+1]: return JP[k]+(0.75-cv[k])/(cv[k+1]-cv[k])
    return np.nan
NB=200
M['season']=pd.to_datetime(M.peak_date).dt.month.map(lambda m:['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'][m-1])
allidx=np.arange(len(M)); pool=curve(allidx)
curves={b:curve(np.where(M.ab==b)[0]) for b in lab}
RC={}; TC={}
for rg in 'EW':
    idx=np.where(M.reg==rg)[0]; bs=np.array([curve(boot_idx(idx)) for _ in range(NB)])
    RC[rg]=(curve(idx),np.percentile(bs,2.5,0),np.percentile(bs,97.5,0)); t=np.array([tclear(c) for c in bs]); TC[rg]=(tclear(RC[rg][0]),np.nanpercentile(t,2.5),np.nanpercentile(t,97.5))
print('tclear E/W',TC)
rows=[]
for rg in 'EW':
    for b in lab:
        idx=np.where((M.ab==b)&(M.reg==rg))[0]
        if len(idx)<30: continue
        lg=np.array([M.L.values[boot_idx(idx)].mean() for _ in range(NB)])
        rows.append(dict(reg=rg,b=b,n=len(idx),A=np.median(M.drain_area_km2.values[idx]),Lmean=M.L.values[idx].mean(),Ll=np.percentile(lg,2.5),Lh=np.percentile(lg,97.5)))
R=pd.DataFrame(rows); print(R.round(2).to_string())
# region x season cloud curves, cross-fitted between two random halves of the sites
fr=np.random.default_rng(3); us=M.site_no.unique(); fold=dict(zip(us,fr.integers(0,2,len(us)))); fo=M.site_no.map(fold).values
jj=M.L.clip(0,4).values+2; pred=np.full(len(M),np.nan); TS=[]
for (rg,se),d in M.groupby(['reg','season']):
    ii=M.index.get_indexer(d.index); TS.append(dict(reg=rg,season=se,n=len(ii),tclear=tclear(curve(ii))))
    for f in (0,1):
        cv=curve(ii[fo[ii]!=f]); te=ii[fo[ii]==f]; pred[te]=cv[jj[te]]
M['pred']=pred; TS=pd.DataFrame(TS); print(TS.round(2).to_string())
G=[]
for (b,se,rg),d in M.groupby(['ab','season','reg'],observed=True):
    if len(d)<30: continue
    G.append(dict(b=b,season=se,reg=rg,n=len(d),obs=d.b0.mean()/d.clim_b.mean(),pred=(d.pred*d.clim_b).sum()/d.clim_b.sum()))
G=pd.DataFrame(G); r=np.corrcoef(G.obs,G.pred)[0,1]; rmse=np.sqrt(((G.obs-G.pred)**2).mean()); print('cells',len(G),'r',round(r,3),'rmse',round(rmse,3))
R.to_csv('fig/fig3_lag_by_area_region.csv',index=False); G.to_csv('fig/fig3_cells.csv',index=False); TS.to_csv('fig/fig3_tclear_region_season.csv',index=False)
pd.DataFrame({'jp':JP,'pooled':pool,**{'E':RC['E'][0],'W':RC['W'][0]},**{b:curves[b] for b in lab}}).to_csv('fig/fig3_curves.csv',index=False)
# ---------------- figure ----------------
cols=['#b2182b','#d6604d','#f4a582','#a6a6a6','#92c5de','#4393c3','#2166ac']
fig=plt.figure(figsize=(7.4,5.6)); gs=fig.add_gridspec(2,3,height_ratios=[1.05,1])
ax=[fig.add_subplot(gs[0,:2]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2])]
# a: map of mean crest lag after storm day
p=M.groupby('site_no').agg(lat=('lat','first'),lon=('lon','first'),L=('L','mean'),A=('drain_area_km2','first')).sort_values('L')
sc=ax[0].scatter(p.lon,p.lat,c=p.L,s=3+2.2*np.log10(p.A),cmap='viridis_r',vmin=0,vmax=3,lw=0,rasterized=True)
ax[0].set_aspect(1/np.cos(np.deg2rad(38))); ax[0].set_xlim(-125,-66.5); ax[0].set_ylim(24.5,49.8)
for sp in ['top','right']: ax[0].spines[sp].set_visible(True)
ax[0].set_xticks([-120,-110,-100,-90,-80,-70]); ax[0].set_xticklabels([f'{-v}°W' for v in [-120,-110,-100,-90,-80,-70]]); ax[0].set_yticks([30,35,40,45]); ax[0].set_yticklabels([f'{v}°N' for v in [30,35,40,45]])
ax[0].annotate('',xy=(-70,32.2),xytext=(-70,28.6),arrowprops=dict(arrowstyle='-|>,head_width=0.3,head_length=0.6',color='k',lw=1)); ax[0].text(-70,27.0,'N',ha='center',fontsize=7,fontweight='bold')
kd=111.32*np.cos(np.deg2rad(27)); ax[0].plot([-123.5,-123.5+500/kd],[26,26],'k-',lw=2); ax[0].text(-123.5+250/kd,26.6,'500 km',ha='center',fontsize=6.5)
cb=fig.colorbar(sc,ax=ax[0],shrink=0.75,pad=0.01,extend='max'); cb.set_label('Crest lag after storm day (days)',fontsize=7)
# b: lag distribution by area
T=pd.crosstab(M.ab,M.L.clip(0,4),normalize='index').reindex(lab)
lc=plt.cm.viridis_r(np.linspace(0.05,0.95,5)); left=np.zeros(len(lab))
for k in range(5):
    ax[1].barh(np.arange(len(lab)),T[k].values,left=left,color=lc[k],height=0.75,lw=0,label=['0','1','2','3','≥4'][k]); left+=T[k].values
for i,v in enumerate(T[0].values):
    if v>=0.12: ax[1].text(v/2,i,f'{v*100:.0f}%',ha='center',va='center',fontsize=6,color='w' if v>0.12 else 'k')
ax[1].set_yticks(np.arange(len(lab))); ax[1].set_yticklabels(labp,fontsize=6.5); ax[1].invert_yaxis(); ax[1].set_xlim(0,1); ax[1].set_xlabel('Share of crests'); ax[1].set_ylabel('Drainage area (km²)')
ax[1].legend(title='Lag (d)',ncol=5,frameon=False,fontsize=6,title_fontsize=6.5,loc='lower center',bbox_to_anchor=(0.5,1.0),handlelength=1,columnspacing=0.6)
# c: storm-aligned clear-sky curves
ax[2].axvspan(-0.5,0.5,color='#d9d9d9',alpha=0.6,lw=0)
for b,c in zip(lab,cols): ax[2].plot(JP,curves[b],'-',color=c,lw=0.8,alpha=0.9)
for rg,ls,nm in [('E','-','East of 100°W'),('W','--','West of 100°W')]:
    ax[2].fill_between(JP,RC[rg][1],RC[rg][2],color='k',alpha=0.15,lw=0); ax[2].plot(JP,RC[rg][0],'k',ls=ls,lw=1.6,label=nm)
ax[2].axhline(1,color='k',lw=0.5,ls=':'); ax[2].axhline(0.75,color='k',lw=0.5,ls='--'); ax[2].set_xlabel('Days after storm day'); ax[2].set_ylabel('Clear sky / monthly normal'); ax[2].set_xticks(range(-2,8)); ax[2].set_ylim(0,1.45)
ax[2].text(0,1.37,'Storm',ha='center',fontsize=6.5); ax[2].legend(frameon=False,fontsize=6.3,loc='lower right')
# d: two timescales vs area
for rg,mk,ls in [('E','o','-'),('W','^','--')]:
    d=R[R.reg==rg]; ax[3].fill_between(d.A,d.Ll,d.Lh,color='#1b7837',alpha=0.18,lw=0); ax[3].plot(d.A,d.Lmean,ls,color='#1b7837',lw=1.1)
    for x,y,b in zip(d.A,d.Lmean,d.b): ax[3].plot(x,y,mk,color=cols[lab.index(b)],mec='k',mew=0.3,ms=4,zorder=5)
    t,tl,th=TC[rg]; ax[3].axhspan(tl,th,color='#4d4d4d',alpha=0.18,lw=0); ax[3].axhline(t,color='#4d4d4d',ls=ls,lw=1.1)
    ax[3].text(9,t+0.06,f'Cloud clearing, {"East" if rg=="E" else "West"}',fontsize=6.3,color='#4d4d4d')
ax[3].text(9,0.15,'Flood crest',fontsize=6.3,color='#1b7837')
from matplotlib.lines import Line2D
ax[3].legend([Line2D([],[],color='k',ls='-',marker='o',ms=3.5,mfc='w',mew=0.6),Line2D([],[],color='k',ls='--',marker='^',ms=3.5,mfc='w',mew=0.6)],['East','West'],frameon=False,fontsize=6.3,loc='center left',bbox_to_anchor=(0.0,0.86))
ax[3].set_xscale('log'); ax[3].set_xlim(7,4e4); ax[3].set_xlabel('Drainage area (km²)'); ax[3].set_ylabel('Days after storm day'); ax[3].set_ylim(0,3.0)
# e: cross-fitted prediction
for b,c in zip(lab,cols):
    for rg,mk in [('E','o'),('W','^')]:
        d=G[(G.b==b)&(G.reg==rg)]; ax[4].scatter(d.pred,d.obs,s=8+d.n/14,marker=mk,color=c,edgecolor='k',lw=0.3,alpha=0.9,label=labp[lab.index(b)] if rg=='E' else None)
ax[4].plot([0,1.8],[0,1.8],'k--',lw=0.6); ax[4].set_xlim(0,1.65); ax[4].set_ylim(0,1.65); ax[4].set_aspect('equal')
ax[4].set_xlabel('Predicted crest-day clear sky / normal'); ax[4].set_ylabel('Observed'); ax[4].text(0.06,1.58,f'r = {r:.2f}\nRMSE = {rmse:.2f}\nn = {len(G)} groups',fontsize=6.3,va='top')
ax[4].legend([Line2D([],[],ls='',marker='o',ms=4,mfc=c,mec='k',mew=0.3) for c in cols]+[Line2D([],[],ls='',marker='^',ms=4,mfc='w',mec='k',mew=0.5)],labp+['West'],frameon=False,fontsize=5.5,loc='lower right',handletextpad=0.1,borderaxespad=0.1,labelspacing=0.25)
for a,l in zip(ax,'abcde'): a.text(-0.08 if l=='a' else -0.27,1.03,l,transform=a.transAxes,fontsize=10,fontweight='bold')
fig.tight_layout(w_pad=0.8,h_pad=1.0); fig.savefig('fig/fig3_mechanism.png',dpi=200); print('ok')
