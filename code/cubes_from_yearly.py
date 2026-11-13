# Rebuild the daily GOES-16 clear-sky cubes data/terra.npy, data/aqua.npy and data/days.npy
# (2018-2024, 0.1 degree, uint8) from the yearly files written by cloudcube.py.
import numpy as np, glob
fs = sorted(glob.glob('data/clear_20[12][0-9].npz'))
Z = [np.load(f) for f in fs]
np.save('data/terra.npy', np.concatenate([z['terra'] for z in Z]))
np.save('data/aqua.npy', np.concatenate([z['aqua'] for z in Z]))
np.save('data/days.npy', np.concatenate([z['days'] for z in Z]))
print(len(fs), 'years,', sum(len(z['days']) for z in Z), 'days')
