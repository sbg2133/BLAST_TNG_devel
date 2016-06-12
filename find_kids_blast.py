import numpy as np
import os
import matplotlib.pyplot as plt

path = '/home/muchacho/readout_data/iqstream_penn_May16/vna_sweeps/scaled2'
bb_freqs, freq_step = np.linspace(-200.0e6, 200.0e6, 500, retstep = True)

def openStored(path):
	files = sorted(os.listdir(path))
	lo_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
	I_list = [os.path.join(path, filename) for filename in files if filename.startswith('I')]
	Q_list = [os.path.join(path, filename) for filename in files if filename.startswith('Q')]
	chan_I = np.array([np.load(filename) for filename in I_list])
	chan_Q = np.array([np.load(filename) for filename in Q_list])
	return lo_freqs, chan_I, chan_Q

def filter_trace():
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
	return chan_freqs, mags

def lowpass_cosine( y, tau, f_3db, width, padd_data=True):
	import numpy as nm
        # padd_data = True means we are going to symmetric copies of the data to the start and stop
	# to reduce/eliminate the discontinuities at the start and stop of a dataset due to filtering
	#
	# False means we're going to have transients at the start and stop of the data

	# kill the last data point if y has an odd length
	if nm.mod(len(y),2):
		y = y[0:-1]

	# add the weird padd
	# so, make a backwards copy of the data, then the data, then another backwards copy of the data
	if padd_data:
		y = nm.append( nm.append(nm.flipud(y),y) , nm.flipud(y) )

	# take the FFT
        import scipy
        import scipy.fftpack
	ffty=scipy.fftpack.fft(y)
	ffty=scipy.fftpack.fftshift(ffty)

	# make the companion frequency array
	delta = 1.0/(len(y)*tau)
	nyquist = 1.0/(2.0*tau)
	freq = nm.arange(-nyquist,nyquist,delta)
	# turn this into a positive frequency array
	pos_freq = freq[(len(ffty)/2):]

	# make the transfer function for the first half of the data
	i_f_3db = min( nm.where(pos_freq >= f_3db)[0] )
	f_min = f_3db - (width/2.0)
	i_f_min = min( nm.where(pos_freq >= f_min)[0] )
	f_max = f_3db + (width/2);
	i_f_max = min( nm.where(pos_freq >= f_max)[0] )

	transfer_function = nm.zeros(len(y)/2)
	transfer_function[0:i_f_min] = 1
	transfer_function[i_f_min:i_f_max] = (1 + nm.sin(-nm.pi * ((freq[i_f_min:i_f_max] - freq[i_f_3db])/width)))/2.0
	transfer_function[i_f_max:(len(freq)/2)] = 0

	# symmetrize this to be [0 0 0 ... .8 .9 1 1 1 1 1 1 1 1 .9 .8 ... 0 0 0] to match the FFT
	transfer_function = nm.append(nm.flipud(transfer_function),transfer_function)

	# apply the filter, undo the fft shift, and invert the fft
	filtered=nm.real(scipy.fftpack.ifft(scipy.fftpack.ifftshift(ffty*transfer_function)))

	# remove the padd, if we applied it
	if padd_data:
		filtered = filtered[(len(y)/3):(2*(len(y)/3))]

	# return the filtered data
        return filtered
	

sweep_step = 5.0 # kHz
smoothing_scale = 1500.0 # kHz
peak_threshold = 0.4 # mag units
spacing_threshold = 50.0 # kHz

chan_freqs,mags = filter_trace()
filtermags = lowpass_cosine( mags, sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))

plt.ion()
plt.figure(1)
plt.clf()
plt.plot(chan_freqs,mags,'b',label='#nofilter')
plt.plot(chan_freqs,filtermags,'g',label='Filtered')
plt.legend()

plt.figure(2)
plt.clf()
plt.plot(chan_freqs,mags-filtermags,'b')
ilo = np.where( (mags-filtermags) < -1.0*peak_threshold)[0]
plt.plot(chan_freqs[ilo],mags[ilo]-filtermags[ilo],'r*')

edges = np.where(np.diff(ilo) > (spacing_threshold/sweep_step))[0]
edges = np.append(np.array([0]),edges)
centers = np.round((edges[0:-2] + edges[1:-1])/2.0).astype('int')
ind_kids = ilo[centers]

# find actual peaks near these centers
for i in xrange(len(ind_kids)):
	# get some data nearby
	num = np.round((spacing_threshold/sweep_step)/2.0).astype('int')
	nearby = (mags-filtermags)[(ind_kids[i]-num):(ind_kids[i]+num)]
	ihi = np.where(np.abs(nearby) == np.max(np.abs(nearby)))[0] - num
	ind_kids[i] = ind_kids[i] + ihi

print len(edges)

plt.figure(4)
plt.clf()
plt.plot(chan_freqs,mags,'g')
plt.plot(chan_freqs[ind_kids],mags[ind_kids],'r*')

# list of kid frequencies
target_freqs = chan_freqs[ind_kids]
