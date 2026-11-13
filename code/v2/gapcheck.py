# Report the blank horizontal and vertical bands of each figure (mm at the 170 mm print width), to audit panel spacing
import sys, glob, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def bands(white, mm, lo=2.5):
    out = []; s = None
    for i, w in enumerate(list(white) + [False]):
        if w and s is None: s = i
        elif not w and s is not None:
            if (i - s) * mm >= lo and s > 0 and i < len(white): out.append((round(s * mm, 1), round(i * mm, 1), round((i - s) * mm, 1)))
            s = None
    return out
for f in sorted(glob.glob('fig/v2/Fig*.png')):
    if 'test' in f or (len(sys.argv) > 1 and not any(a in f for a in sys.argv[1:])): continue
    im = np.asarray(Image.open(f).convert('L').resize((1700, int(round(Image.open(f).size[1] * 1700 / Image.open(f).size[0])))))
    mm = 170 / im.shape[1]; ink = im < 245
    rows = ink.mean(1) < 0.001                      # a blank band may hold a panel letter only if it is very small, so use a strict threshold
    print(f.split('/')[-1], 'H', round(im.shape[0] * mm, 1), 'mm | blank rows (from, to, height):', bands(rows, mm), '| top margin', round(np.argmax(~rows) * mm, 1), 'bottom', round(np.argmax(~rows[::-1]) * mm, 1))
