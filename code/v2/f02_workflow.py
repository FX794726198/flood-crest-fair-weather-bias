# Fig. 1  Why optical satellites miss the flood crests of small basins, and the four steps of the study
from cstyle import *
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Polygon, Ellipse, Circle, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
sys.path.insert(0, HERE)
from f02_terrain import *
from f02_icons import *

SMALL = '#C8312B'; LARGE = '#2166AC'; STEP = '#2F4256'; FLOODW = '#3C7DBA'; CHANW = '#2F6EA8'
H = 138.5; fig = figure(H)
T = build()
LANDC = LinearSegmentedColormap.from_list('l', ['#DCE3BA', '#C2D29A', '#A3BE80', '#89A96D', '#76955F', '#7E8B62', '#9A9677', '#BDB6A2'])

def colours(main_level, trib_level, dull):
    Z = T['Z']; zn = np.clip((Z - 0.2) / 2.0, 0, 1); rgb = LANDC(zn)[..., :3]
    gy_, gx_ = np.gradient(Z, 6 / NY, 10 / NX); sl = np.hypot(gx_, gy_)
    rock = (np.clip((sl - 1.6) / 1.4, 0, 1) * np.clip((Z - 0.9) / 0.6, 0, 1))[..., None]
    rgb = rgb * (1 - 0.55 * rock) + np.array(to_rgb('#A89F8C')) * 0.55 * rock
    ls = LightSource(azdeg=315, altdeg=42)
    sh = ls.shade_rgb(rgb, Z, vert_exag=1.6, blend_mode='soft', dx=10 / NX, dy=6 / NY)
    hm = Z - zc(XX); wm = (hm < main_level) & (np.abs(T['d']) < 1.8)
    ht = Z - T['zt']; yq = YY - T['yj']
    bump = np.clip((yq - 1.10) / 0.35, 0, 1) * np.clip((2.60 - yq) / 0.5, 0, 1)                 # the flash flood spreads over the village flat
    tl = trib_level * np.maximum(0.35 + 0.65 * bump, 0.5 * np.clip((yq - 2.15) / 0.3, 0, 1))
    wt = (ht < tl) & T['inval']
    Zd = np.where(wm, np.maximum(Z, zc(XX) + main_level), Z); Zd = np.where(wt, np.maximum(Zd, T['zt'] + tl), Zd)
    depth = np.where(wm, main_level - hm, 0) + np.where(wt & ~wm, tl - ht, 0)
    # soft shoreline: blend water into land over a thin band of height above the water surface
    am = np.clip((main_level + 0.006 - hm) / 0.012, 0, 1) * (np.abs(T['d']) < 1.8)
    at = np.clip((tl + 0.006 - ht) / 0.012, 0, 1) * T['win']
    a = np.maximum(am, at)[..., None]
    wc = np.array(to_rgb('#8DBCE3')); wd = np.array(to_rgb('#2F6EA8'))
    f = np.clip(depth / 0.08, 0, 1)[..., None]
    sh = sh * (1 - a) + (wc * (1 - f) + wd * f) * a
    if dull: sh = sh * 0.80 + np.array([0.30, 0.33, 0.38]) * 0.20 * 0.9
    return Zd, np.clip(sh, 0, 1)

from matplotlib.colors import to_rgb
def block(l, t, w, h, main_level, trib_level, dull):
    ax = ax_mm(fig, l, t, w, h, projection='3d', computed_zorder=False)
    Zd, sh = colours(main_level, trib_level, dull)
    ax.plot_surface(XX, YY, Zd, facecolors=sh, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, rasterized=True)
    zb = -0.55
    soil = ['#D6C6A3', '#C2AE86'] if not dull else ['#C4B697', '#B0A07E']
    fx = np.r_[x, x[::-1]]; fz = np.r_[Zd[0], np.full(NX, zb)]
    ax.add_collection3d(Poly3DCollection([list(zip(fx, np.zeros_like(fx), fz))], facecolor=soil[0], edgecolor='#7E6B4F', lw=0.4))
    ry = np.r_[y, y[::-1]]; rz = np.r_[Zd[:, -1], np.full(NY, zb)]
    ax.add_collection3d(Poly3DCollection([list(zip(np.full_like(ry, 10.0), ry, rz))], facecolor=soil[1], edgecolor='#7E6B4F', lw=0.4))
    ax.set_proj_type('ortho'); ax.view_init(elev=ELEV, azim=AZIM)
    ax._zoom = ZOOM; ax.set_box_aspect((10, 6, 2.0), zoom=ZOOM); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_zlim(zb, 2.6)
    ax.patch.set_alpha(0)
    return ax, Zd

ELEV, AZIM, ZOOM = 30, -70, 1.0
B1, Z1 = block(-5, 5, 95, 60, 0.02, 0.115, True)
B2, Z2 = block(80, 5, 95, 60, 0.086, -0.016, False)          # small basin back in its channel
ov = ax_mm(fig, 0, 0, W, H); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.set_axis_off(); ov.patch.set_alpha(0)

def _P(ax, X, Y, Z):
    xs, ys, _ = proj3d.proj_transform(np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z), ax.get_proj())
    d = ax.transData.transform(np.c_[xs, ys]); dpi = fig.dpi
    return np.c_[d[:, 0] / dpi * 25.4, H - d[:, 1] / dpi * 25.4]

def fit(ax, Zd, box):
    """scale and move a block so that its outline fills box = (l, t, w, h) in mm"""
    for _ in range(3):
        fig.canvas.draw()
        sub = (slice(None, None, 8), slice(None, None, 8))
        pts = np.r_[_P(ax, XX[sub].ravel(), YY[sub].ravel(), Zd[sub].ravel()), _P(ax, [0, 10, 10, 0], [0, 0, 6, 6], [-0.55] * 4)]
        l, t = pts.min(0); r, b = pts.max(0); k = min(box[2] / (r - l), box[3] / (b - t))
        ax._zoom *= k; ax.set_box_aspect((10, 6, 2.0), zoom=ax._zoom)
        fig.canvas.draw(); pts = np.r_[_P(ax, XX[sub].ravel(), YY[sub].ravel(), Zd[sub].ravel()), _P(ax, [0, 10, 10, 0], [0, 0, 6, 6], [-0.55] * 4)]
        l, t = pts.min(0); r, b = pts.max(0)
        dx = box[0] + (box[2] - (r - l)) / 2 - l; dy = box[1] + (box[3] - (b - t)) / 2 - t
        pos = ax.get_position(); ax.set_position([pos.x0 + dx / W, pos.y0 - dy / H, pos.width, pos.height])
BOX1 = (3, 14, 79, 42); BOX2 = (89, 14, 79, 42)
fit(B1, Z1, BOX1); fit(B2, Z2, BOX2)
fig.canvas.draw()

def P(ax, X, Y, Z):
    xs, ys, _ = proj3d.proj_transform(np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z), ax.get_proj())
    d = ax.transData.transform(np.c_[xs, ys]); dpi = fig.dpi
    return np.c_[d[:, 0] / dpi * 25.4, H - d[:, 1] / dpi * 25.4]

def zat(Zd, X, Y):
    i = int(round(Y / 6 * (NY - 1))); j = int(round(X / 10 * (NX - 1))); return Zd[i, j]

def label(x, y, s, ha='left', va='center', fs=6.6, c=INK, bold=False, z=40):
    ov.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, fontweight='bold' if bold else 'normal', zorder=z, path_effects=[pe.withStroke(linewidth=1.6, foreground='white')])

# ======================================================================= a: two views of the same valley
def label(x, y, s, ha='left', va='center', fs=6.5, c=INK, bold=False, z=40, halo=True):
    ov.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, fontweight='bold' if bold else 'normal', zorder=z, linespacing=1.0,
            path_effects=[pe.withStroke(linewidth=1.8, foreground='white')] if halo else None)

def leader(p0, p1, c=INK, lw=0.5):
    ov.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=lw, zorder=39, solid_capstyle='round')
    ov.add_patch(Circle(p0, 0.45, fc=c, ec='white', lw=0.3, zorder=39.5))

def ground(ax, Zd, X, Y): return _P(ax, X, Y, zat(Zd, X, Y))[0]

# the same village (small-basin valley flat) and town (large-river floodplain) in both blocks
VILLAGE = [(1.62, -0.27), (1.75, 0.24), (1.93, -0.20), (2.05, 0.27)]          # (distance up the tributary from the main channel, offset across it)
TOWN = [(5.75, -1.02), (6.2, -1.22), (6.55, -0.96), (6.95, -1.18), (7.35, -1.0), (7.8, -1.2)]   # (x, distance in front of the main channel)
def settlements(ax, Zd):
    yj0 = float(yr(xt(1.9)))
    pts = [(float(xt(yj0 + a)) + b, yj0 + a) for a, b in VILLAGE] + [(X_, float(yr(X_)) + d_) for X_, d_ in TOWN]
    for X_, Y_ in sorted(pts, key=lambda p: -p[1]):
        q = ground(ax, Zd, X_, Y_); house(ov, q[0], q[1], s=0.72, z=24)

# block 1: crest day of the small basin
yt_ = 3.9; pc = ground(B1, Z1, float(xt(yt_)), yt_)
cb = (pc[0] + 1.0, 19.5)                                                    # storm-cloud base centre
gx = np.linspace(0.2, 9.8, 80); gp = np.array([ground(B1, Z1, xx, 3.4) for xx in gx])
rain(ov, cb[0] - 10.5, cb[0] + 10.5, cb[1] + 0.4, lambda xm: np.interp(xm, gp[:, 0], gp[:, 1]), n=55, slant=-2.2, color='#4F6F98', z=17, seed=2, lw=0.32, alpha=0.6)
cloud(ov, cb[0] + 23, cb[1] - 4.5, s=0.5, kind='storm', z=18.5)
cloud(ov, cb[0], cb[1], s=0.86, kind='storm', z=19)
sat1 = (58, 5.0); tip = satellite(ov, *sat1, s=0.58, rot=-12)
hit = (cb[0] + 8.0, cb[1] - 10.5)
ov.add_patch(Polygon([tip, (hit[0] - 2.6, hit[1] - 1.2), (hit[0] + 2.0, hit[1] + 1.6)], closed=True, fc='#97A3B4', ec='none', alpha=0.40, zorder=18.8))
label(sat1[0] + 5.0, sat1[1] + 0.3, 'View blocked\nby cloud', ha='left', va='center', fs=6.2, c=INK2)
settlements(B1, Z1)
yv = float(yr(xt(1.9))) + 1.50; pt = ground(B1, Z1, float(xt(yv)), yv); leader(pt, (pt[0] - 12, pt[1] + 17.5), c=SMALL)
label(pt[0] - 12, pt[1] + 18.1, 'Small basin\nat its crest', ha='center', va='top', fs=6.5, c=SMALL, bold=True)
pm = ground(B1, Z1, 4.3, float(yr(4.3))); leader(pm, (pm[0] + 2, pm[1] + 11.8))
label(pm[0] + 2, pm[1] + 12.4, 'Large river low', ha='center', va='top', fs=6.5)

# block 2: crest of the large river three days later
sun(ov, 118, 8.0, r=2.2, z=19)
cloud(ov, 136, 9.5, s=0.42, kind='fair', z=19); cloud(ov, 162, 14.5, s=0.26, kind='fair', z=19)
sat2 = (157, 4.6); tip2 = satellite(ov, *sat2, s=0.58, rot=-12)
X2 = 6.75; Y2 = float(yr(X2)) - 1.03; pr = ground(B2, Z2, X2, Y2)                     # the flooded town
ov.add_patch(Polygon([tip2, (pr[0] - 7.0, pr[1] + 0.6), (pr[0] + 7.0, pr[1] - 0.6)], closed=True, fc='#F4CD6E', ec='none', alpha=0.30, zorder=17.5))
ov.add_patch(Ellipse(pr, 14.4, 3.2, angle=-4, fc='none', ec='#D99A2B', lw=0.7, zorder=23))
label(sat2[0] - 5.5, sat2[1] + 0.3, 'Clear view', ha='right', va='center', fs=6.2, c=INK2)
settlements(B2, Z2)
yu = float(yr(xt(1.9))) + 2.45; pt2 = ground(B2, Z2, float(xt(yu)), yu); leader(pt2, (pt2[0] - 12, pt2[1] - 11))
label(pt2[0] - 12, pt2[1] - 11.6, 'Small basin\ndrained', ha='center', va='bottom', fs=6.5)
pm2 = ground(B2, Z2, 3.6, float(yr(3.6)) - 0.7); leader(pm2, (pm2[0] - 5, pm2[1] + 10), c=LARGE)
label(pm2[0] - 5, pm2[1] + 10.6, 'Large river\nat its crest', ha='center', va='top', fs=6.5, c=LARGE, bold=True)

ov.text(6.0, 0.9, 'Day 0', fontsize=7.5, fontweight='bold', va='top', color=INK)
ov.text(15.3, 0.9, 'Storm overhead', fontsize=7.5, va='top', color=INK)
ov.text(92.0, 0.9, 'Day 3', fontsize=7.5, fontweight='bold', va='top', color=INK)
ov.text(101.3, 0.9, 'Skies clear', fontsize=7.5, va='top', color=INK)
letter(fig, 'a', 1, 0.4)

# ======================================================================= b: crest lag against cloud clearing
TB = 59.5; LB = 15; WB = 151
tt = np.linspace(-0.6, 5.2, 600)
clr = np.where(tt < 0, 0.80, 1 - 0.78 * np.exp(-np.clip(tt, 0, None) / 0.85))
qs = np.where(tt < 0, 0.0, np.where(tt < 0.3, (np.clip(tt, 0, None) / 0.3) ** 1.6, np.exp(-(tt - 0.3) / 0.75)))
tp = 3.2; qa = 10; ql = np.where(tt > 0, (np.clip(tt, 1e-6, None) / tp) ** qa * np.exp(qa * (1 - tt / tp)), 0.0)
sk = ax_mm(fig, LB, TB - 1.5, WB, 6.5)
cmap_sky = LinearSegmentedColormap.from_list('sky', ['#5F6B7B', '#A7B3C2', '#E6EFF8'])
sk.imshow(clr[None, :], cmap=cmap_sky, vmin=0.2, vmax=1.0, extent=(tt[0], tt[-1], 0, 1), aspect='auto', interpolation='bicubic')
sk.set_xlim(tt[0], tt[-1]); sk.set_ylim(0, 1); sk.set_xticks([]); sk.set_yticks([])
for sp in sk.spines.values(): sp.set_visible(False)
looks = [k + f for k in range(0, 6) for f in (0.44, 0.56)]
first = None
for t_ in looks:
    if t_ > tt[-1]: continue
    c_ = np.interp(t_, tt, clr); ok = c_ >= 0.6
    sk.plot(t_, 0.32, 'o', ms=3.2, mfc='#4F8FB0' if ok else 'white', mec='#1F3B57' if ok else '#7A7A7A', mew=0.5, zorder=4)
sk.text(0.05, 0.74, 'Storm cloud', fontsize=6.2, color='white', va='center', ha='left', fontweight='bold')
sk.text(1.40, 0.74, 'Cloud clears', fontsize=6.2, color=INK, va='center', ha='left')
fig.text((LB - 1.5) / W, 1 - (TB + 1.75) / H, 'Sky at\noverpasses', fontsize=6.3, ha='right', va='center', linespacing=0.95)

hb = ax_mm(fig, LB, TB + 6.2, WB, 24)
hb.fill_between(tt, 0, qs, color=SMALL, alpha=0.12, lw=0); hb.plot(tt, qs, color=SMALL, lw=1.3)
hb.fill_between(tt, 0, ql, color=LARGE, alpha=0.12, lw=0); hb.plot(tt, ql, color=LARGE, lw=1.3)
hb.plot(0.3, 1.0, marker='v', ms=4.2, color=SMALL, mec='white', mew=0.4, zorder=5); hb.plot(tp, 1.0, marker='v', ms=4.2, color=LARGE, mec='white', mew=0.4, zorder=5)
hb.axvline(0, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
f1 = 1.44; y1 = float(np.interp(f1, tt, qs)); f2 = 3.44; y2 = float(np.interp(f2, tt, ql))
for fx_, fy_, col in [(f1, y1, SMALL), (f2, y2, LARGE)]:
    hb.plot([fx_, fx_], [fy_, 1.43], color='#1F3B57', lw=0.6, ls=(0, (1.5, 1.5)), zorder=2, clip_on=False)
    hb.plot(fx_, fy_, 'o', ms=4.0, mfc='#4F8FB0', mec='#1F3B57', mew=0.6, zorder=6)
hb.annotate('First clear look:\n22% of crest', xy=(f1, y1), xytext=(f1 + 0.12, 0.60), fontsize=6.5, color=SMALL, va='bottom', ha='left', linespacing=1.0,
            arrowprops=dict(arrowstyle='-', lw=0.5, color=SMALL, shrinkA=0.5, shrinkB=2.5))
hb.text(f2 + 0.12, 1.03, 'First clear look:\n97% of crest', fontsize=6.5, color=LARGE, va='bottom', ha='left', linespacing=1.0)
hb.text(0.40, 1.03, 'Small basin', fontsize=6.6, color=SMALL, va='bottom', ha='left', fontweight='bold')
hb.text(tp - 0.12, 1.03, 'Large river', fontsize=6.6, color=LARGE, va='bottom', ha='right', fontweight='bold')
hb.set_xlim(tt[0], tt[-1]); hb.set_ylim(0, 1.32); hb.set_yticks([0, 0.5, 1]); hb.set_xticks(range(0, 6))
hb.set_xlabel('Days after the storm', labelpad=1.5); hb.set_ylabel('Flow / crest flow')
letter(fig, 'b', 1, TB - 2.5)
for j_, (mf, me, nm) in enumerate([('#4F8FB0', '#1F3B57', 'clear'), ('white', '#7A7A7A', 'cloudy')]):
    lx = 4.05 + j_ * 0.52
    sk.plot(lx, 0.74, 'o', ms=3.0, mfc=mf, mec=me, mew=0.5, zorder=5); sk.text(lx + 0.07, 0.74, nm, fontsize=6.0, va='center', ha='left', color=INK2, zorder=5)

# ======================================================================= c: four steps of the study
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
def wrap(t, fs, wmm, bold=False):
    fp = FontProperties(family='sans-serif', weight='bold' if bold else 'normal'); out, line = [], ''
    for w_ in t.split(' '):
        cand = (line + ' ' + w_).strip()
        if TextPath((0, 0), cand, size=fs, prop=fp).get_extents().width * 25.4 / 72 > wmm and line: out.append(line); line = w_
        else: line = cand
    return '\n'.join(out + [line])
TC = 97.5; GAP = 4.6; M0 = 0.8; CW = (W - 2 * M0 - 3 * GAP) / 4; CL = [M0 + k * (CW + GAP) for k in range(4)]; PADX = 2.4; TW = CW - 2 * PADX
STEPS = [('Measure', 'Is the crest seen?', 'Small basins see 0.43 of normal clear sky, large 0.88'),
         ('Explain', 'Why are small basins missed?', 'Timing predicts the gradient, r\u00a0=\u00a00.90–0.94'),
         ('Correct with TWCC', 'What did the missed crest hold?', 'Crest depths within 25%: 34%\u00a0→\u00a062%'),
         ('Design', 'How often must satellites look?', 'Cloud-free 4 h looks: mean 80% of small-basin crest extent')]
for k, (ttl, q, res) in enumerate(STEPS):
    x0 = CL[k]
    ov.add_patch(FancyBboxPatch((x0, TC), CW, 40.6, boxstyle='round,pad=0,rounding_size=1.6', fc='#F3F5F7', ec='none', zorder=1))
    ov.add_patch(Circle((x0 + 4.0, TC + 4.0), 2.35, fc=STEP, ec='none', zorder=3))
    ov.text(x0 + 4.0, TC + 4.05, str(k + 1), ha='center', va='center', fontsize=7.2, color='white', fontweight='bold', zorder=4)
    ov.text(x0 + 7.6, TC + 4.0, ttl, ha='left', va='center', fontsize=7.6, fontweight='bold', color=INK, zorder=4)
    ov.text(x0 + PADX, TC + 8.7, q, ha='left', va='center', fontsize=6.5, color=INK2, zorder=4)
    ov.plot([x0 + PADX, x0 + CW - PADX], [TC + 33.4, TC + 33.4], color=GREY_L, lw=0.5, zorder=4)
    ov.text(x0 + PADX, TC + 34.3, wrap(res, 6.4, TW, bold=True), ha='left', va='top', fontsize=6.4, color=INK, fontweight='bold', zorder=4, linespacing=1.05)
    if k < 3: arrow(ov, (x0 + CW + 0.7, TC + 21.0), (x0 + CW + GAP - 0.7, TC + 21.0), color=STEP, lw=1.0, hw=3.0, hl=3.0, z=5)
letter(fig, 'c', 1, TC - 3.2)
ys0, ys1 = TC + 11.8, TC + 29.6                  # schematic band

# 1 Measure: the same gauged basin on crest day and on a normal day of that month
from scipy.ndimage import gaussian_filter as _gf
NC, cs_ = 7, 1.95; gw = NC * cs_; gx0 = [CL[0] + PADX + 0.6, CL[0] + PADX + 0.6 + gw + 5.6]; gy0 = ys1 - gw - 0.4
rng = np.random.default_rng(7); base_f = _gf(rng.standard_normal((NC, NC)), 1.0); base_f = (base_f - base_f.mean()) / base_f.std()
tt_b = np.linspace(0, 2 * np.pi, 200); rb = 1 + 0.18 * np.sin(2 * tt_b + 0.5) + 0.10 * np.cos(3 * tt_b)
for g, (gx, thr, ttl_) in enumerate(zip(gx0, (1.0, -0.9), ('Crest day', 'Normal'))):
    for i in range(NC):
        for j in range(NC):
            cloudy = base_f[i, j] < thr
            ov.add_patch(Rectangle((gx + j * cs_, gy0 + i * cs_), cs_, cs_, fc='#FFFFFF' if cloudy else '#C6DAA8', ec='#C5CCD5', lw=0.3, zorder=5))
    cx_, cy_ = gx + gw * 0.5, gy0 + gw * 0.47
    ov.plot(cx_ + 0.30 * gw * rb * np.cos(tt_b) * 0.95, cy_ + 0.30 * gw * rb * np.sin(tt_b) * 1.1, color=SMALL, lw=0.9, zorder=6)
    gauge(ov, cx_ + 0.05 * gw, cy_ + 0.30 * gw * 1.1 * 1.0 + 0.2, s=0.75, z=7)
    ov.text(gx + gw / 2, gy0 - 1.1, ttl_, ha='center', va='bottom', fontsize=6.0, color=INK, zorder=6)
ov.text(gx0[0] + gw + 2.8, gy0 + gw / 2, 'vs', ha='center', va='center', fontsize=6.2, color=INK2, zorder=6)
for j_, (fc_, nm) in enumerate([('#C6DAA8', 'Clear'), ('#FFFFFF', 'Cloud')]):
    lx = gx0[0] + j_ * 13.0
    ov.add_patch(Rectangle((lx, ys1 + 0.35), 1.7, 1.7, fc=fc_, ec='#C5CCD5', lw=0.3, zorder=6))
    ov.text(lx + 2.5, ys1 + 1.2, nm, ha='left', va='center', fontsize=5.9, color=INK2, zorder=6)

# 2 Explain: storm-aligned clearing curve read at each crest lag
x0 = CL[1] + PADX + 3.0; xw = TW - 3.4; y0_, y1_ = ys1 - 0.6, ys0 + 0.2
ov.annotate('', xy=(x0 + xw, y0_), xytext=(x0, y0_), arrowprops=dict(arrowstyle='-|>,head_width=0.16,head_length=0.32', lw=0.6, color=INK), zorder=6)
ov.annotate('', xy=(x0, y1_), xytext=(x0, y0_), arrowprops=dict(arrowstyle='-|>,head_width=0.16,head_length=0.32', lw=0.6, color=INK), zorder=6)
u = np.linspace(0, 1, 100); cy_ = y0_ - (y0_ - y1_ - 3.2) * (1 - 0.82 * np.exp(-u / 0.22))
ov.plot(x0 + 0.5 + u * (xw - 2.5), cy_, color='#56677D', lw=1.2, zorder=6)
ov.plot([x0, x0 + xw - 1.5], [y1_ + 3.2, y1_ + 3.2], color=GREY, lw=0.5, ls=(0, (3, 2)), zorder=5)
ov.text(x0 + 1.2, y1_ + 2.6, 'normal', ha='left', va='bottom', fontsize=5.9, color=INK2, zorder=6)
for uu, col, nm, dx_, dy_ in [(0.06, SMALL, 'small', 1.6, 0.0), (0.66, LARGE, 'large', 1.3, 2.4)]:
    xx = x0 + 0.5 + uu * (xw - 2.5); yy = float(np.interp(uu, u, cy_))
    ov.plot([xx, xx], [yy, y0_], color=col, lw=0.6, ls=(0, (1.5, 1.2)), zorder=6)
    ov.add_patch(Circle((xx, yy), 0.85, fc=col, ec='white', lw=0.4, zorder=7))
    ov.text(xx + dx_, yy + dy_, nm, ha='left', va='center', fontsize=5.9, color=col, zorder=7, fontweight='bold')
ov.text(x0 + xw / 2, y0_ + 0.9, 'Crest lag after the storm', ha='center', va='top', fontsize=5.9, color=INK2, zorder=6)
ov.text(x0 - 0.9, (y0_ + y1_) / 2 + 0.5, 'Clear sky', ha='right', va='center', rotation=90, fontsize=5.9, color=INK2, zorder=6)

# 3 Correct: the timing-weighted crest correction, the weight of a late image falls with the wait, fastest in small basins
x0 = CL[2] + PADX + 3.0; xw = TW - 3.4; y0_, y1_ = ys1 - 0.6, ys0 + 4.4
ov.text(CL[2] + PADX, ys0 + 0.2, r'Crest = image$^{\,w}$ × prior$^{\,1-w}$', ha='left', va='top', fontsize=6.0, color=INK, zorder=6)
ov.annotate('', xy=(x0 + xw, y0_), xytext=(x0, y0_), arrowprops=dict(arrowstyle='-|>,head_width=0.16,head_length=0.32', lw=0.6, color=INK), zorder=6)
ov.annotate('', xy=(x0, y1_), xytext=(x0, y0_), arrowprops=dict(arrowstyle='-|>,head_width=0.16,head_length=0.32', lw=0.6, color=INK), zorder=6)
u = np.linspace(0, 1, 100); tau = u * 4.5
for Tw, col, nm in [(7.65, LARGE, 'large'), (1.11, SMALL, 'small')]:
    wv_ = 1 / (1 + ((tau + 1) / Tw) ** 1.26); yy_ = y0_ - (y0_ - y1_ - 0.6) * wv_
    ov.plot(x0 + 0.5 + u * (xw - 2.5), yy_, color=col, lw=1.2, zorder=6)
    ov.text(x0 + xw - 1.6, yy_[-1] - 0.9, nm, ha='right', va='bottom', fontsize=5.9, color=col, fontweight='bold', zorder=7)
ov.text(x0 + xw / 2, y0_ + 0.9, 'Wait after the crest', ha='center', va='top', fontsize=5.9, color=INK2, zorder=6)
ov.text(x0 - 0.9, (y0_ + y1_) / 2 + 0.3, 'Image weight', ha='right', va='center', rotation=90, fontsize=5.9, color=INK2, zorder=6)

# 4 Design: crest duration against revisit interval
x0 = CL[3] + PADX + 8.6; xw = TW - 8.6; ty = ys1 + 0.2
ov.annotate('', xy=(x0 + xw + 0.6, ty), xytext=(x0, ty), arrowprops=dict(arrowstyle='-|>,head_width=0.16,head_length=0.32', lw=0.6, color=INK), zorder=6)
ov.text(x0 - 1.0, ty, 'Time', ha='right', va='center', fontsize=5.9, color=INK2, zorder=6)
sb0, sb1 = 0.22, 0.31
ov.add_patch(FancyBboxPatch((x0 + sb0 * xw, ty - 3.3), (sb1 - sb0) * xw, 2.0, boxstyle='round,pad=0,rounding_size=0.5', fc=SMALL, ec='none', zorder=6))
ov.add_patch(FancyBboxPatch((x0 + 0.06 * xw, ty - 6.3), 0.86 * xw, 2.0, boxstyle='round,pad=0,rounding_size=0.5', fc=LARGE, ec='none', zorder=6))
ov.text(x0 - 1.0, ty - 2.3, 'Small', ha='right', va='center', fontsize=5.9, color=SMALL, zorder=6, fontweight='bold')
ov.text(x0 - 1.0, ty - 5.3, 'Large', ha='right', va='center', fontsize=5.9, color=LARGE, zorder=6, fontweight='bold')
ov.text(x0 + sb1 * xw + 1.0, ty - 2.3, 'crest duration', ha='left', va='center', fontsize=5.9, color=INK2, zorder=6)
for dt_, yy, nm in [(0.5, ty - 10.2, 'Daily'), (0.083, ty - 14.4, 'Every 4 h')]:
    for t_ in np.arange(0.035, 1.0, dt_):
        hitS = sb0 <= t_ <= sb1
        ov.plot([x0 + t_ * xw] * 2, [yy - 1.0, yy + 1.0], color=SMALL if hitS else '#56677D', lw=1.0 if hitS else 0.6, zorder=6)
    ov.text(x0 - 1.0, yy, nm, ha='right', va='center', fontsize=5.9, color=INK2, zorder=6)
ov.text(x0 - 1.0, ty - 17.4, 'Revisit', ha='right', va='center', fontsize=5.9, color=INK, zorder=6, fontweight='bold')

save(fig, 'Fig02_workflow', check=False, tight=False)
