"""Shared style for the cloud-hidden flood crest figures.
All figures are drawn at the final printed width of the Word layout (170 mm text width), so font sizes are final sizes.
Axes are placed in millimetres from the top-left corner, so panels of one row share their top and bottom edges and
panels of one column share their left edge. Panel letters sit at a common height per row."""
import os, sys, json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import overlap_check            # noqa: E402  (also applies a base style; ours overrides it below)
OUT = os.path.join(ROOT, 'fig', 'v2'); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 7,
    'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'axes.titlesize': 7.5,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
    'mathtext.bf': 'Liberation Sans:bold', 'mathtext.default': 'regular', 'axes.unicode_minus': True,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45,
    'xtick.major.size': 2.8, 'ytick.major.size': 2.8, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8, 'axes.labelpad': 2.2, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.45, 'legend.borderaxespad': 0.25,
    'legend.labelspacing': 0.28, 'legend.columnspacing': 0.9, 'lines.linewidth': 1.2, 'lines.markersize': 3.6,
    'lines.markeredgewidth': 0.5, 'errorbar.capsize': 0, 'pdf.fonttype': 42, 'savefig.pad_inches': 0.0,
    'hatch.linewidth': 0.5, 'axes.axisbelow': True, 'figure.dpi': 100})
FS = 7.0; FSS = 6.4; FL = 10        # annotation, small annotation, panel letter (pt)
W = 170.0                           # figure width (mm)
MM = 1 / 25.4
HALO = [pe.withStroke(linewidth=2.0, foreground='white')]

# ---------------------------------------------------------------- colours (one meaning per colour in every figure)
LAB = ['<30', '30–100', '100–300', '300–1k', '1k–3k', '3k–10k', '>10k']
LABK = ['<30', '30-100', '100-300', '300-1k', '1k-3k', '3k-10k', '>10k']          # keys used in the data tables
BINS = [0, 30, 100, 300, 1000, 3000, 10000, 1e7]
XC = np.array([13.0, 55.0, 173.0, 548.0, 1730.0, 5480.0, 31600.0])               # plotting position of each class on a log area axis
ACOL = ['#8E0F17', '#C8312B', '#EE7A4A', '#F5B971', '#8FBCDB', '#3F84BC', '#164F86']   # small = warm, large = cool
ACOL_D = dict(zip(LABK, ACOL))
INK = '#262626'; INK2 = '#5A5A5A'; GREY = '#8C8C8C'; GREY_L = '#C9C9C9'; GREY_LL = '#ECECEC'
LAND = '#E6E6E6'; LAND_EDGE = '#9A9A9A'; SEA = '#FFFFFF'
CLOUD = '#7F8FA6'; CLEAR = '#E8B04A'; WATER = '#2F7FC1'; WATER_L = '#A9CDEB'
OBS = '#B2182B'; PRED = '#2166AC'                        # observed vs predicted / first look vs rebuilt
FIRST = '#B2182B'; REBUILT = '#2166AC'
EAST = '#1B7837'; WEST = '#762A83'
REG = {'namerica': '#0072B2', 'camelsgb': '#009E73', 'lamah': '#CC79A7', 'camelsbr': '#E69F00', 'camelscl': '#D55E00', 'camelsaus': '#8C6D1F'}
REGN = {'namerica': 'North America', 'camelsgb': 'Great Britain', 'lamah': 'Central Europe', 'camelsbr': 'Brazil', 'camelscl': 'Chile', 'camelsaus': 'Australia'}
SEAS = {'DJF': '#3A6FB0', 'MAM': '#5BA05B', 'JJA': '#D9822B', 'SON': '#8E5A9E'}
SEASN = {'DJF': 'Winter', 'MAM': 'Spring', 'JJA': 'Summer', 'SON': 'Autumn'}

def aclass(a):
    return pd.cut(a, BINS, labels=LABK)

# ---------------------------------------------------------------- layout
def figure(H):
    fig = plt.figure(figsize=(W * MM, H * MM)); fig._H = H; return fig

def ax_mm(fig, l, t, w, h, **k):
    H = fig._H; return fig.add_axes([l / W, 1 - (t + h) / H, w / W, h / H], **k)

def letter(fig, s, x, t, fs=FL):
    """panel letter with its top-left corner at (x, t) mm"""
    fig.text(x / W, 1 - t / fig._H, s, fontsize=fs, fontweight='bold', va='top', ha='left', color='k')

def area_axis(ax, label=True, lim=(7, 60000), minor=True):
    ax.set_xscale('log'); ax.set_xlim(*lim)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {10: '10', 100: '10$^2$', 1000: '10$^3$', 10000: '10$^4$'}.get(int(round(v)), '')))
    if not minor: ax.xaxis.set_minor_locator(NullLocator())
    if label: ax.set_xlabel('Drainage area (km$^2$)')

def class_ticks(ax, short=True, rot=0, label=True):
    """categorical x axis with the seven area classes"""
    ax.set_xticks(range(7)); ax.set_xticklabels(['<30', '30–\n100', '100–\n300', '300–\n1k', '1k–\n3k', '3k–\n10k', '>10k'] if short else LAB, rotation=rot, linespacing=0.9)
    ax.set_xlim(-0.5, 6.5)
    if label: ax.set_xlabel('Drainage area (km$^2$)')

def band(ax, x, lo, hi, c, a=0.18, **k):
    ax.fill_between(x, lo, hi, color=c, alpha=a, lw=0, **k)

def ref(ax, y=1.0, **k):
    ax.axhline(y, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0, **k)

def stat(ax, s, x, y, ha='left', va='top', fs=FSS, color=INK, **k):
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=color, **k)

def spines(ax, keep=('left', 'bottom')):
    for k_, sp in ax.spines.items(): sp.set_visible(k_ in keep)

def frame(ax):
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)

# ---------------------------------------------------------------- maps
_NE = {}
def _rings(name):
    if name not in _NE:
        g = json.load(open(f'data/{name}.geojson')); out = []
        for f in g['features']:
            geo = f['geometry']
            if geo is None: continue
            cs = geo['coordinates'] if geo['type'] in ('MultiPolygon', 'MultiLineString') else [geo['coordinates']]
            for p in cs:
                if geo['type'].endswith('Polygon'): out.append(np.array(p[0]))
                else: out.append(np.array(p))
        _NE[name] = out
    return _NE[name]

def basemap(ax, extent, states=False, land=LAND, edge=LAND_EDGE, lw=0.3, sea=SEA, z=0):
    from matplotlib.collections import PolyCollection, LineCollection
    ax.set_facecolor(sea)
    ax.add_collection(PolyCollection(_rings('ne_50m_land'), facecolor=land, edgecolor=edge, linewidth=lw, zorder=z))
    if states: ax.add_collection(LineCollection([r for r in _rings('ne_50m_states')], colors='white', linewidths=0.35, zorder=z + 0.5))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

def geo_ticks(ax, xt, yt, fs=6.5):
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([(f'{abs(v)}°W' if v < 0 else (f'{v}°E' if v > 0 else '0°')) for v in xt], fontsize=fs)
    ax.set_yticklabels([(f'{abs(v)}°S' if v < 0 else (f'{v}°N' if v > 0 else '0°')) for v in yt], fontsize=fs)
    ax.tick_params(length=2, pad=1.2); frame(ax)

def north(ax, x, y, size=0.07, color='k'):
    """north arrow at axes fraction (x, y)"""
    ax.annotate('', xy=(x, y + size), xytext=(x, y), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45', color=color, lw=0.9))
    ax.text(x, y - 0.01, 'N', transform=ax.transAxes, ha='center', va='top', fontsize=6.3, fontweight='bold', color=color)

def scalebar(ax, lon0, lat0, km, lat_ref=None, label=None, color='k', fs=6.2, h=None):
    lat_ref = lat0 if lat_ref is None else lat_ref
    dl = km / (111.32 * np.cos(np.deg2rad(lat_ref)))
    ax.plot([lon0, lon0 + dl], [lat0, lat0], color=color, lw=1.8, solid_capstyle='butt', zorder=8)
    yr = ax.get_ylim(); dy = (yr[1] - yr[0]) * 0.025 if h is None else h
    ax.text(lon0 + dl / 2, lat0 + dy, label or f'{km:g} km', ha='center', va='bottom', fontsize=fs, color=color, zorder=8)

def cbar(fig, l, t, w, h, mappable, label=None, orient='horizontal', ticks=None, fmt=None, fs=6.5):
    cax = ax_mm(fig, l, t, w, h)
    cb = fig.colorbar(mappable, cax=cax, orientation=orient, ticks=ticks, format=fmt)
    cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=fs, length=1.8, width=0.4, pad=1.2)
    if label:
        if orient == 'horizontal': cb.set_label(label, fontsize=fs + 0.3, labelpad=1.5)
        else: cb.set_label(label, fontsize=fs + 0.3, labelpad=2)
    return cb

# ---------------------------------------------------------------- output and checks
def _blank_bands(fig):
    """full-width blank bands of the rendered figure as (from, to) in mm from the top, with the first and last inked rows"""
    fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    rows = (buf.min(2) < 245).any(1); n = len(rows); mm = fig._H / n; idx = np.where(rows)[0]
    out = []; s_ = None
    for i in range(idx[0], idx[-1] + 1):
        if not rows[i] and s_ is None: s_ = i
        elif rows[i] and s_ is not None: out.append((s_ * mm, i * mm)); s_ = None
    return out, idx[0] * mm, (n - 1 - idx[-1]) * mm

def _placed_by_parent(ax):
    """True for inset axes that follow a parent axes. A colourbar axes carries a locator too, but it reads its own
    position, so it has to be moved like any other axes."""
    loc = ax.get_axes_locator()
    if loc is None: return False
    if type(loc).__name__ == '_ColorbarAxesLocator': return getattr(loc, '_orig_locator', None) is not None
    return True

def compact(fig, gap=2.5, bottom=1.2, name=''):
    """Close the blank bands between panel rows to `gap` mm and trim the bottom margin to `bottom` mm.
    Everything below a band moves up as a block, so panels keep their size in millimetres and nothing can come to overlap."""
    H = fig._H; bands, top, bot = _blank_bands(fig)
    cut = [(a_, b_, b_ - a_ - gap) for a_, b_ in bands if b_ - a_ > gap + 0.25]; trim = max(0.0, bot - bottom) if bot > bottom + 0.25 else 0.0
    if not cut and not trim: return
    R = fig.canvas.get_renderer(); px = fig.bbox.height
    def mm_c(art):                                           # centre of an artist in mm from the top
        bb = art.get_window_extent(R); return (1 - (bb.y0 + bb.y1) / 2 / px) * H
    def up(c): return sum(e for a_, b_, e in cut if c > (a_ + b_) / 2)
    for ax in fig.axes:
        p_ = ax.get_position(); t0, t1 = (1 - p_.y1) * H, (1 - p_.y0) * H
        if any(t0 < a_ and t1 > b_ for a_, b_, _ in cut): print(name, 'not compacted: an axes spans a blank band'); return
    Hn = H - sum(e for *_, e in cut) - trim
    for ax in fig.axes:
        if _placed_by_parent(ax): continue
        d = up((1 - (ax.get_position().y0 + ax.get_position().y1) / 2) * H)
        for which, orig in (('original', True), ('active', False)):
            p_ = ax.get_position(original=orig); t_ = (1 - p_.y1) * H - d; h_ = p_.height * H
            ax.set_position([p_.x0, 1 - (t_ + h_) / Hn, p_.width, h_ / Hn], which=which)
    for t in fig.texts:
        x_, y_ = t.get_position(); d = up(mm_c(t)); t.set_position((x_, 1 - ((1 - y_) * H - d) / Hn))
    for lg in fig.legends:
        d = up(mm_c(lg)); bb = lg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        t_ = (1 - bb.y1) * H - d; h_ = bb.height * H
        lg.set_bbox_to_anchor((bb.x0, 1 - (t_ + h_) / Hn, bb.width, h_ / Hn), transform=fig.transFigure)
    assert not (fig.patches or fig.lines or fig.artists or fig.images), 'figure-level artists are not handled by compact()'
    fig.set_size_inches(W * MM, Hn * MM); fig._H = Hn
    b2, _, bot2 = _blank_bands(fig)
    print(name, f'compacted {H:.1f} -> {Hn:.1f} mm; row gaps now', [round(b_ - a_, 1) for a_, b_ in b2 if b_ - a_ > 1.5], 'bottom', round(bot2, 1))

def save(fig, name, check=True, ignore=(), tight=True):
    if tight: compact(fig, name=name)
    fig.savefig(f'{OUT}/{name}.png', dpi=600); fig.savefig(f'{OUT}/{name}.pdf')
    if check:
        issues = [i for i in overlap_check(fig) if not any(s in str(i) for s in ignore)]
        print(name, 'layout issues:', issues if issues else 'none')
    plt.close(fig)

def boot_ratio(df, num, den, site='site_no', nb=300, seed=0):
    rng = np.random.default_rng(seed); g = df.groupby(site).indices; keys = np.array(list(g)); r = []
    for _ in range(nb):
        ii = np.concatenate([g[k] for k in rng.choice(keys, len(keys))]); d = df.iloc[ii]; r.append(d[num].sum() / d[den].sum())
    return np.percentile(r, [2.5, 97.5])
