"""Drawing helpers for the Fig. 1 schematic. All coordinates are millimetres on an overlay axes with y increasing downwards."""
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon, Circle, Ellipse, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb

def _circles_path(cs):
    verts, codes = [], []
    for (x, y, r) in cs:
        p = Path.circle((x, y), r)
        verts.append(p.vertices); codes.append(p.codes)
    return Path(np.concatenate(verts), np.concatenate(codes))

def _grad(ax, path, top, bot, z, alpha=1.0, extent=None):
    bb = path.get_extents(); x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    cm = LinearSegmentedColormap.from_list('g', [top, bot])
    g = np.linspace(0, 1, 64)[:, None] * np.ones((1, 4))
    im = ax.imshow(g, cmap=cm, extent=(x0, x1, y1, y0), origin='upper', aspect='auto', zorder=z, alpha=alpha, interpolation='bicubic')
    pp = PathPatch(path, fc='none', ec='none', transform=ax.transData); ax.add_patch(pp); im.set_clip_path(pp)
    return im

def cloud(ax, cx, cy, s=1.0, kind='fair', z=20, alpha=1.0):
    """cumulus centred at (cx, cy) with flat base at cy; s scales width (base width about 18*s mm)"""
    if kind == 'storm':
        cs = [(-9, -2.2, 3.4), (-5.2, -4.2, 4.4), (-0.6, -6.2, 5.6), (4.6, -5.0, 4.8), (8.8, -2.8, 3.6), (1.5, -10.5, 4.6),
              (-3.0, -9.5, 3.8), (5.0, -9.2, 3.6), (0.2, -13.5, 3.8), (-11.5, -1.4, 2.4), (11.3, -1.6, 2.4)]
        top, bot = '#AEB6C2', '#4E5867'
    else:
        cs = [(-5.5, -1.6, 2.4), (-2.4, -3.2, 3.2), (1.6, -3.6, 3.4), (4.8, -2.0, 2.5), (-0.4, -1.4, 2.6), (7.0, -1.1, 1.6), (-7.6, -0.9, 1.5)]
        top, bot = '#FFFFFF', '#C9D2DD'
    def un(dr, base):
        q = _circles_path([(cx + s * a, cy + s * b, s * r + dr) for a, b, r in cs])
        v = q.vertices.copy(); v[:, 1] = np.minimum(v[:, 1], base); return Path(v, q.codes)
    edge = '#6F7A8A' if kind == 'storm' else '#A9B4C2'
    ax.add_patch(PathPatch(un(0.22, cy + 0.22), fc=edge, ec='none', zorder=z - 0.05, alpha=alpha))
    p = un(0, cy); _grad(ax, p, top, bot, z, alpha)
    return p

def anvil(ax, cx, cy, s=1.0, z=19):
    """flat anvil top of a cumulonimbus whose tower top is at (cx, cy)"""
    xs = np.array([-16, -10, -4, 4, 12, 19, 21, 14, 4, -6, -14, -17]) * s + cx
    ys = np.array([0.6, -0.6, -1.4, -1.6, -1.3, -0.5, 0.4, 1.4, 2.4, 2.2, 1.5, 0.9]) * s + cy
    poly = Path(np.c_[xs, ys], closed=False)
    t = np.linspace(0, 1, len(xs))
    pp = Path(np.r_[np.c_[xs, ys], [[xs[0], ys[0]]]], [Path.MOVETO] + [Path.LINETO] * (len(xs) - 1) + [Path.CLOSEPOLY])
    _grad(ax, pp, '#C7CED8', '#8C96A5', z)
    ax.add_patch(PathPatch(pp, fc='none', ec='#7D8796', lw=0.35, zorder=z + 0.1))

def rain(ax, x0, x1, ytop, ybot_fn, n=60, slant=1.2, color='#5F7FA6', z=18, seed=1, lw=0.35, alpha=0.75):
    """slanted rain streaks from the cloud base to the ground, fading downwards"""
    rng = np.random.default_rng(seed); segs, cols = [], []
    rgb = to_rgb(color); k = 8
    for _ in range(n):
        x = rng.uniform(x0, x1); yb = ybot_fn(x + slant) if callable(ybot_fn) else ybot_fn
        y = ytop + rng.uniform(0.0, 0.25) * (yb - ytop); L = yb - ytop
        pts = [(x + slant * (y + (yb - y) * j / k - ytop) / max(L, 1e-6), y + (yb - y) * j / k) for j in range(k + 1)]
        for j in range(k):
            segs.append([pts[j], pts[j + 1]]); cols.append((*rgb, alpha * (1 - 0.85 * j / k)))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=z, capstyle='butt'))

def sun(ax, x, y, r=2.2, z=20):
    for k in range(12):
        a = k * np.pi / 6; ax.plot([x + 1.45 * r * np.cos(a), x + 1.95 * r * np.cos(a)], [y + 1.45 * r * np.sin(a), y + 1.95 * r * np.sin(a)],
                                    color='#E9A23B', lw=0.7, solid_capstyle='round', zorder=z)
    p = Path.circle((x, y), r); _grad(ax, p, '#FFE08A', '#F0A43A', z)
    ax.add_patch(Circle((x, y), r, fc='none', ec='#D98A2B', lw=0.4, zorder=z + 0.1))

def satellite(ax, x, y, s=1.0, z=30, rot=-18):
    """satellite centred at (x, y); s=1 gives about 11 mm span"""
    c, sn = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    R = lambda P: np.array([[x + s * (px * c - py * sn), y + s * (px * sn + py * c)] for px, py in P])
    # panels
    for side in (-1, 1):
        x0 = 1.35 * side; x1 = 5.6 * side
        P = [(x0, -1.05), (x1, -1.05), (x1, 1.05), (x0, 1.05)]
        ax.add_patch(Polygon(R(P), closed=True, fc='#24406E', ec='#16294A', lw=0.4, zorder=z))
        for k in range(1, 4):
            xx = x0 + (x1 - x0) * k / 4; seg = R([(xx, -1.05), (xx, 1.05)]); ax.plot(seg[:, 0], seg[:, 1], color='#6F8FC2', lw=0.3, zorder=z + 0.1)
        seg = R([(x0, 0), (x1, 0)]); ax.plot(seg[:, 0], seg[:, 1], color='#6F8FC2', lw=0.3, zorder=z + 0.1)
        seg = R([(0.9 * side, 0), (x0, 0)]); ax.plot(seg[:, 0], seg[:, 1], color='#555555', lw=0.6, zorder=z - 0.1)
    # body with a lit and a shaded face
    ax.add_patch(Polygon(R([(-1.0, -1.3), (1.0, -1.3), (1.0, 1.3), (-1.0, 1.3)]), closed=True, fc='#D9B456', ec='#8E6F22', lw=0.4, zorder=z + 0.2))
    ax.add_patch(Polygon(R([(0.15, -1.3), (1.0, -1.3), (1.0, 1.3), (0.15, 1.3)]), closed=True, fc='#B8902F', ec='none', zorder=z + 0.25))
    # sensor pointing down
    ax.add_patch(Polygon(R([(-0.55, 1.3), (0.55, 1.3), (0.4, 2.0), (-0.4, 2.0)]), closed=True, fc='#7A7A7A', ec='#4A4A4A', lw=0.3, zorder=z + 0.2))
    return R([(0, 2.0)])[0]

def house(ax, x, y, s=1.0, z=25, wall='#F4EEE2', roof='#A9442F'):
    """house standing on (x, y); s=1 is about 2.2 mm wide"""
    w, h, r = 1.1 * s, 1.0 * s, 0.85 * s
    ax.add_patch(Polygon([(x - w, y), (x + w, y), (x + w, y - h), (x - w, y - h)], closed=True, fc=wall, ec='#4D3B30', lw=0.35, zorder=z))
    ax.add_patch(Polygon([(x - w * 1.25, y - h), (x, y - h - r), (x + w * 1.25, y - h)], closed=True, fc=roof, ec='#4D3B30', lw=0.35, zorder=z + 0.1))
    ax.add_patch(Rectangle((x - 0.25 * s, y - 0.55 * s), 0.5 * s, 0.55 * s, fc='#6B5444', ec='none', zorder=z + 0.1))

def gauge(ax, x, y, s=1.0, z=26, color='#1F3B57'):
    """water-level symbol: inverted triangle above two short lines; (x, y) is the triangle tip"""
    ax.add_patch(Polygon([(x - 1.0 * s, y - 1.5 * s), (x + 1.0 * s, y - 1.5 * s), (x, y)], closed=True, fc=color, ec='white', lw=0.4, zorder=z))
    for k, L in ((0.45, 1.0), (0.9, 0.6)):
        ax.plot([x - L * s, x + L * s], [y + k * s, y + k * s], color=color, lw=0.6, zorder=z, solid_capstyle='round')

def arrow(ax, p0, p1, color='#33475B', lw=0.9, hw=2.2, hl=2.6, z=30, rad=0.0, ls='-'):
    a = FancyArrowPatch(p0, p1, arrowstyle=f'-|>,head_width={hw / 10:.2f},head_length={hl / 10:.2f}', mutation_scale=10, color=color, lw=lw,
                        zorder=z, connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, linestyle=ls)
    ax.add_patch(a); return a
