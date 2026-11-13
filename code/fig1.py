import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'axes.spines.top':False,'axes.spines.right':False})
s=pd.read_parquet('data/main_sample.parquet'); rng=np.random.default_rng(0)
bins=[0,30,100,300,1000,3000,10000,1e7]; s['abin']=pd.cut(s.drain_area_km2,bins,labels=['<30','30–100','100–300','300–1k','1k–3k','3k–10k','>10k'])
lab=list(s.abin.cat.categories); xc=np.arange(len(lab))
def boot(df,fn,nb=300):
    sites=df.site_no.unique(); idx={k:v for k,v in df.groupby('site_no').indices.items()}
    r=[]
    for _ in range(nb):
        ss=rng.choice(sites,len(sites)); ii=np.concatenate([idx[k] for k in ss]); r.append(fn(df.iloc[ii]))
    return np.percentile(r,[2.5,97.5])
rows=[]
for b in lab:
    d=s[s.abin==b]
    r0=d.b0.mean()/d.clim_b.mean(); ci0=boot(d,lambda x:x.b0.mean()/x.clim_b.mean())
    rc=d.cap.mean()/d.cap_clim.mean(); cic=boot(d,lambda x:x.cap.mean()/x.cap_clim.mean())
    q=d.q_at_first.dropna(); rows.append(dict(b=b,n=len(d),r0=r0,r0l=ci0[0],r0h=ci0[1],rc=rc,rcl=cic[0],rch=cic[1],q25=q.quantile(.25),q50=q.median(),q75=q.quantile(.75)))
R=pd.DataFrame(rows); R.to_csv('fig/fig1_bins.csv',index=False); print(R.round(2).to_string())
fig=plt.figure(figsize=(7.4,5.6)); gs=fig.add_gridspec(2,3,height_ratios=[1.05,1],width_ratios=[1,1,1])
ax=[fig.add_subplot(gs[0,:2]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2])]
# a: map of sites, colour = median flow fraction at first clear look
p=s.groupby('site_no').agg(lat=('lat','first'),lon=('lon','first'),q=('q_at_first','median'),A=('drain_area_km2','first'),n=('q_at_first','size'))
p=p[p.n>=2].sort_values('q',ascending=False)
sc=ax[0].scatter(p.lon,p.lat,c=p.q,s=4,cmap='YlOrRd_r',vmin=0,vmax=1,lw=0,rasterized=True)
ax[0].set_aspect(1/np.cos(np.deg2rad(38))); ax[0].set_xlim(-125,-66.5); ax[0].set_ylim(24.5,49.8)
for sp in ['top','right']: ax[0].spines[sp].set_visible(True)
ax[0].set_xticks([-120,-110,-100,-90,-80,-70]); ax[0].set_xticklabels([f'{-v}°W' for v in [-120,-110,-100,-90,-80,-70]]); ax[0].set_yticks([30,35,40,45]); ax[0].set_yticklabels([f'{v}°N' for v in [30,35,40,45]])
ax[0].annotate('',xy=(-68.5,48.8),xytext=(-68.5,45.2),arrowprops=dict(arrowstyle='-|>,head_width=0.3,head_length=0.6',color='k',lw=1)); ax[0].text(-68.5,43.6,'N',ha='center',fontsize=7,fontweight='bold')
kd=111.32*np.cos(np.deg2rad(27)); ax[0].plot([-123.5,-123.5+500/kd],[26,26],'k-',lw=2); ax[0].text(-123.5+250/kd,26.6,'500 km',ha='center',fontsize=6.5)
cb=fig.colorbar(sc,ax=ax[0],shrink=0.75,pad=0.01); cb.set_label('Flow at first clear look / peak flow',fontsize=7)
# b: ratio vs offset
OFF=list(range(-3,11)); cols=['#b2182b','#ef8a62','#67a9cf','#2166ac']
for (nm,sel),c in zip([('<100 km²',s.drain_area_km2<100),('100–1,000',s.drain_area_km2.between(100,1000)),('1,000–10,000',s.drain_area_km2.between(1000,1e4)),('>10,000',s.drain_area_km2>1e4)],cols):
    y=[np.nanmean(s.loc[sel,f'b{o}'])/np.nanmean(s.loc[sel,'clim_b']) for o in OFF]; ax[1].plot(OFF,y,'o-',ms=2.5,lw=1.2,color=c,label=nm)
ax[1].axhline(1,color='k',lw=0.5,ls=':'); ax[1].axvline(0,color='k',lw=0.5,ls='--'); ax[1].set_xlabel('Days from flood crest'); ax[1].set_ylabel('Clear sky / monthly normal'); ax[1].legend(frameon=False,fontsize=6.5,loc='lower right'); ax[1].set_ylim(0.3,1.15)
# c: crest-day clear ratio by area
ax[2].errorbar(xc,R.r0,yerr=[R.r0-R.r0l,R.r0h-R.r0],fmt='o-',color='#b2182b',ms=3.5,capsize=2,lw=1.2,label='Crest day')
ax[2].errorbar(xc,R.rc,yerr=[R.rc-R.rcl,R.rch-R.rc],fmt='s-',color='#2166ac',ms=3.5,capsize=2,lw=1.2,label='Near-crest capture')
ax[2].axhline(1,color='k',lw=0.5,ls=':'); ax[2].set_xticks(xc); ax[2].set_xticklabels(lab,rotation=45,fontsize=6.5); ax[2].set_xlabel('Drainage area (km²)'); ax[2].set_ylabel('Observed / expected'); ax[2].set_ylim(0.3,1.1); ax[2].legend(frameon=False,fontsize=6.5,loc='lower right')
# d: flow, water depth and flooded area remaining at first clear look
ST=pd.read_parquet('data/stage_first.parquet'); EX=pd.read_parquet('data/extent_first.parquet')
for X in (ST,EX): X['ab']=pd.cut(X.A,bins,labels=lab)
ax[3].fill_between(xc,R.q25,R.q75,color='#ef8a62',alpha=0.25,lw=0); ax[3].plot(xc,R.q50,'o-',color='#b2182b',ms=3.5,lw=1.2,label='Discharge')
ax[3].plot(xc,[ST[ST.ab==b].depth_frac.median() for b in lab],'s-',color='#2166ac',ms=3.5,lw=1.2,label='Water depth')
ax[3].plot(xc,[EX[EX.ab==b].extent_frac.median() for b in lab],'^-',color='#1a9850',ms=3.5,lw=1.2,label='Flooded area (HAND)')
ax[3].set_xticks(xc); ax[3].set_xticklabels(lab,rotation=45,fontsize=6.5); ax[3].set_xlabel('Drainage area (km²)'); ax[3].set_ylabel('Left at first clear look (/ crest)'); ax[3].set_ylim(0,1.05); ax[3].legend(frameon=False,fontsize=6.3,loc='lower right')
# e: flashiness vs crest clear ratio (by inst/daily peak ratio)
s['fl']=pd.qcut(s.inst_to_daily,8)
g=s.groupby('fl',observed=True); xm=g.inst_to_daily.median(); yr=g.b0.mean()/g.clim_b.mean(); yq=g.q_at_first.median()
ax[4].plot(xm,yr,'o-',color='#b2182b',ms=3.5,lw=1.2,label='Crest-day clear sky / normal'); ax[4].plot(xm,yq,'s-',color='#2166ac',ms=3.5,lw=1.2,label='Flow at first clear look / peak')
ax[4].set_xscale('log'); ax[4].set_xticks([1,2,4,8]); ax[4].set_xticklabels(['1','2','4','8']); ax[4].minorticks_off(); ax[4].set_xlabel('Instantaneous / daily-mean peak'); ax[4].set_ylabel('Ratio'); ax[4].set_ylim(0,1.05); ax[4].legend(frameon=False,fontsize=6.2,loc='upper right')
for a,l in zip(ax,'abcde'): a.text(-0.08 if l=='a' else -0.27,1.03,l,transform=a.transAxes,fontsize=10,fontweight='bold')
fig.tight_layout(w_pad=0.8,h_pad=1.0); fig.savefig('fig/fig1_cloud_crest.png',dpi=200); print('ok',len(p))
