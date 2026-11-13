"""Synthetic valley for the Fig. 1 block diagrams: a large river along the front and a small steep tributary basin
draining a mountain range behind it. The range is built from pointed massifs, ridged noise and downslope gullies,
and the tributary valley is carved with concave walls into a small flat floor where a village sits."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
NX, NY = 600, 360
S = NX / 400                                                                   # noise scales are set for a 400-column grid
x = np.linspace(0, 10, NX); y = np.linspace(0, 6, NY); XX, YY = np.meshgrid(x, y)
def yr(x): return 1.83 + 0.30 * np.sin(0.62 * x + 0.4)
def zc(x): return 0.30 - 0.015 * x
def xt(y): return 2.80 + 0.26 * np.sin(1.45 * y + 0.3) + 0.10 * np.sin(3.3 * y + 1.1) - 0.08 * (y - 3.0)

def noise(rng, sig, aniso=1.0):
    n = gaussian_filter(rng.standard_normal(XX.shape), (sig * aniso * S, sig * S), mode='reflect'); return n / n.std()

def fbm(rng, sigmas=(22, 11, 5.5, 2.7, 1.4), gain=0.5, aniso=1.0):
    out = np.zeros(XX.shape); a = 1.0; tot = 0
    for s in sigmas: out += a * noise(rng, s, aniso); tot += a; a *= gain
    return out / tot

def warp(F, rng, amp=6.0, sig=20):
    """domain warp in pixels so that ridges and gullies bend"""
    u = noise(rng, sig) * amp * S; v = noise(rng, sig) * amp * S
    ii, jj = np.mgrid[0:NY, 0:NX].astype(float)
    return map_coordinates(F, [ii + v, jj + u], order=1, mode='reflect')

def build(seed=11):
    rng = np.random.default_rng(seed)
    d = YY - yr(XX)
    zh = zc(XX) + 0.05 + 0.03 * np.abs(d)                                    # floodplain
    db = np.clip(d - 1.25, 0, None); df = np.clip(-d - 1.35, 0, None)
    on = np.clip(db / 0.45, 0, 1)
    foot = 0.26 * (1 - np.exp(-(db / 0.8) ** 1.5))                            # rolling foothills
    yc = 4.82 + 0.18 * np.sin(0.8 * XX + 1.0)                                 # crest line of the range
    across = np.exp(-(np.abs(YY - yc) / np.where(YY < yc, 1.30, 0.85)) ** 1.7)
    # pointed summits along the crest; the small basin heads in the col between the two on the left
    peaks = [(0.4, 0.40, 0.75), (1.75, 1.05, 0.85), (3.75, 0.90, 0.80), (5.6, 0.55, 0.85), (7.45, 1.12, 0.95), (9.35, 0.55, 0.75)]
    pk = sum(h * np.exp(-((np.abs(XX - px) / w) ** 1.3 + (np.abs(YY - 4.82 - 0.18 * np.sin(0.8 * px + 1.0)) / (0.95 * w)) ** 1.3)) for px, h, w in peaks)
    # ridged noise for sharp crests and spurs, warped so that they wander
    rid = np.clip(1 - np.abs(fbm(rng, (16, 8, 4, 2), 0.5)), 0, 1); rid = warp(rid ** 2.2, rng, 5, 24)
    rough = warp(fbm(rng), rng, 4, 18)
    # gullies running downslope (toward the river): stripes of anisotropic noise, troughs where it crosses zero
    gs = warp(noise(rng, 3.2, aniso=3.5), rng, 5, 22); gully = np.exp(-(gs / 0.33) ** 2)
    gs2 = warp(noise(rng, 1.8, aniso=3.5), rng, 2.5, 16); gully2 = np.exp(-(gs2 / 0.30) ** 2)
    flank = np.clip(db / 0.5, 0, 1) * np.clip((yc + 0.1 - YY) / 0.5, 0, 1)
    Zm = on * (foot + 0.80 * across + pk + 0.26 * rid * across + 0.09 * rough)
    Zm = Zm - flank * (0.12 * gully + 0.05 * gully2) * (0.4 + 0.6 * np.clip(Zm / 1.4, 0, 1))
    Z = zh + Zm + 0.10 * df ** 1.2 + 0.03 * rough * np.clip(df, 0, 1)
    Z = Z - 0.07 * np.exp(-(d / 0.10) ** 2)                                  # main channel
    # tributary: concave walls rising from a flat floor that widens where the valley leaves the range (the village)
    dt = XX - xt(YY); yj = yr(xt(YY))
    zt = zc(xt(YY)) + 0.04 + 0.11 * np.clip(YY - yj - 0.5, 0, None) + 0.40 * np.clip(YY - yj - 2.15, 0, None) ** 2.0   # the village flat sits above the large river's flood
    wf = 0.06 + 0.40 * np.exp(-((YY - yj - 1.80) / 0.45) ** 2)                # floor half-width, widest at the village
    wv = 0.40 + 0.45 * np.clip((YY - yj - 0.6) / 2.2, 0, 1) ** 1.5                   # wall length scale
    r = np.clip(np.abs(dt) - wf, 0, None) / wv
    g = np.exp(-r ** 1.15) * np.clip((YY - yj - 0.3) / 0.4, 0, 1) * np.clip((4.95 - YY) / 0.45, 0, 1)
    ztv = np.maximum(zt, zh - 0.012) - 0.035 * np.exp(-(dt / 0.045) ** 2) * np.clip((YY - yj - 1.0) / 0.3, 0, 1)   # floor never below the floodplain
    Zc = ztv + (np.maximum(Z, ztv) - ztv) * (1 - g)
    Z = np.where(g > 0, np.minimum(Z, Zc), Z)
    Z = gaussian_filter(Z, 0.7 * S)
    win = np.clip(1 - (np.abs(dt) - (wf + 0.12)) / 0.12, 0, 1) * np.clip((YY - yj + 0.05) / 0.25, 0, 1) * np.clip((4.80 - YY) / 0.2, 0, 1)
    return dict(Z=Z, d=d, dt=dt, zt=zt, yj=yj, inval=win > 0, win=win, wf=wf)
