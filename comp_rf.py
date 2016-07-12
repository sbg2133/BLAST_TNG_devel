import numpy as np
import matplotlib.pyplot as plt
from read_snaps import FirmwareSnaps
fs = FirmwareSnaps()

rf = np.loadtxt('./10kHZ_RFpower_upconverted.csv', dtype = 'float', skiprows = 108)
min_freq = 300.0e6
max_freq = 1.2e9
rf_freqs = np.linspace(min_freq, max_freq, len(rf))

I, Q = fs.read_accum_snap()
mags = 20*np.log10((np.sqrt(I[:1000]**2 + Q[:1000]**2)))
mags = np.concatenate((mags[len(mags)/2.:],mags[:len(mags)/2.]))
#mags = mags[::-1]


mag_freqs = np.linspace(rf_freqs[178], rf_freqs[623], len(mags))

#offset = np.mean( mags - rf)

plt.ion()
plt.clf()
plt.figure(1)
plt.plot(rf_freqs[178:623], rf[178:623])

plt.plot(mag_freqs , mags - 16.5)
