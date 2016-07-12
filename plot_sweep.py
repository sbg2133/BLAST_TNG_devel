import numpy as np
import os, sys
import matplotlib.pyplot as plt

def open_stored(self, save_path = None):
	files = sorted(os.listdir(save_path))
	sweep_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
	I_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('I')]
	Q_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('Q')]
	Is = np.array([np.load(filename) for filename in I_list])
	Qs = np.array([np.load(filename) for filename in Q_list])
	return lo_freqs, Is, Qs

def filter_trace(bb_freqs):
	lo_freqs, chan_I, chan_Q = openStored(path)
	channels = xrange(np.shape(chan_I)[1])
	sorted_freqs = np.zeros((len(channels),len(lo_freqs)))
	rf_freqs = np.sort(bb_freqs + 750.0e6)
	mag = np.zeros((len(channels),len(lo_freqs)))
	chan_freqs = np.zeros((len(channels),len(lo_freqs)))
	for chan in channels:
		mag[chan] = 20*np.log10(np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2)) 
		chan_freqs[chan] = np.sort(lo_freqs + bb_freqs[chan])
	for chan in channels:
		if chan > 1:
			end_piece = mag[chan - 1][-1]
			offset = mag[chan][0] - end_piece
			mag[chan] = mag[chan] - offset
	mags = np.hstack(mag)
	chan_freqs = np.hstack(chan_freqs)
	return chan_freqs, mags, mag

def plot_trace(bb_freqs, path):
	chan_freqs,mags,mag = filter_trace()

	plt.ion()
	plt.figure(1)
	plt.clf()
	plt.plot(chan_freqs,mags)
	plt.ylabel('dB')
	plt.xlabel('freq')
	return
