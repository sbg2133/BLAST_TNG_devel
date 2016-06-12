import numpy as np
import os
import matplotlib.pyplot as plt

path = '/home/muchacho/readout_data/iqstream_nist_Jan16/vna_sweeps/04'
bb_freqs, freq_step = np.linspace(-255.0e6, 255.0e6, 1000, retstep = True)

def openStored(path):
	files = sorted(os.listdir(path))
	lo_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
	I_list = [os.path.join(path, filename) for filename in files if filename.startswith('I')]
	Q_list = [os.path.join(path, filename) for filename in files if filename.startswith('Q')]
	chan_I = np.array([np.load(filename) for filename in I_list])
	chan_Q = np.array([np.load(filename) for filename in Q_list])
	return lo_freqs, chan_I, chan_Q

def peakFreqs(plot = False):
	lo_freqs, chan_I, chan_Q = openStored(path)
	channels = np.arange(np.shape(chan_I)[1])
	sorted_freqs = np.zeros((len(channels),len(lo_freqs)))
	rf_freqs = np.sort(bb_freqs + 750.0e6)
	mags = np.zeros((len(channels),len(lo_freqs)))
	scaled_mags = np.zeros(np.shape(mags))
	peak_freqs = []	
	Qf = []
	#plt.figure(figsize = (12,12))
	for chan in channels:
		mags[chan] = 20*np.log10(np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2)) 
		diff = 0. - np.mean(mags[chan])
		scaled_mag = diff + mags[chan]
		sorted_freqs[chan] = np.sort(lo_freqs + bb_freqs[chan])
		scaled_mags[chan] = np.abs(scaled_mag)  
	for chan in channels:
		tmp_idx = []
		tmp_idx_2 = []
		tmp_freqs = []
		mags = scaled_mags[chan]/np.max(scaled_mags[chan])
		smags = scaled_mags[chan]
		for i in range(len(smags)):
			if i < 20:
				points = smags[:i + 20]
			elif i > len(smags) - 20:
				points = smags[i - 20:]
			elif (20 <= i <= len(smags) - 20):
				points = smags[i - 20:i + 20]	
			if (smags[i] == np.max(points)) and (i > 2) and (i < len(smags) - 2):	
				idx = np.where(smags == smags[i])[0][0]
				tmp_idx.append(idx)
				peak_freqs.append(sorted_freqs[chan][idx])
		print tmp_idx
		for idx in tmp_idx:
			fc = sorted_freqs[chan][idx]
			fstep = 5
			if idx < fstep:
				left = 0. 
				right = idx + fstep
			elif idx > len(smags) - fstep:
				left = idx - fstep
				right = len(smags) - 1
			elif fstep < idx < len(smags) - fstep:
				left = idx - fstep
				right = idx + fstep
			right_mag = smags[right] 
			left_mag = smags[left]
			avg_mag = (left_mag + right_mag) / 2.
			if (smags[idx]/avg_mag > 1.25):
				print smags[idx]/avg_mag
				tmp_freqs.append(sorted_freqs[chan][idx])
				tmp_idx_2.append(idx)
				if plot:
					plt.plot([sorted_freqs[chan][left], sorted_freqs[chan][right]],[avg_mag, avg_mag],linestyle = '--')
					
				
			#bw = sorted_freqs[chan][idx + step] - sorted_freqs[chan][idx - step] 
			#Q = fc/np.abs(bw)
			#Qf.append(Q)
			#print Q
		print tmp_idx_2
		peak_mags = smags[tmp_idx_2]
		for i in range(len(peak_mags)):
			plt.vlines(tmp_freqs[i],0.,peak_mags[i], linestyle = '--')
		if plot:
			plt.plot(sorted_freqs[chan], smags, c='g')
			plt.scatter(tmp_freqs, peak_mags, c='r')
			plt.title('Chan = ' + str(chan))
			plt.ylabel('Mag (dB)')
			plt.xlabel('Freq (GHz)')
			plt.ion()
			plt.show()	
			raw_input()
			plt.clf()
		Qf = np.array(Qf)
	return  

def filter_trace():
	lo_freqs, chan_I, chan_Q = openStored(path)
	channels = np.arange(np.shape(chan_I)[1])
	sorted_freqs = np.zeros((len(channels),len(lo_freqs)))
	rf_freqs = np.sort(bb_freqs + 750.0e6)
	mag = np.zeros((len(channels),len(lo_freqs)))
	chan_freqs = np.zeros((len(channels),len(lo_freqs)))
	for chan in channels:
		mag[chan] = 20*np.log10(np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2)) 
		chan_freqs[chan] = np.sort(lo_freqs + bb_freqs[chan])
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
	

sweep_step = 2.5 # kHz
smoothing_scale = 1000.0 # kHz
peak_threshold = 2.0 # mag units
spacing_threshold = 50.0 # kHz

chan_freqs,mags = filter_trace()
filtermags = lowpass_cosine( mags, sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
plt.ion()
plt.figure(1)
plt.plot(chan_freqs,mags,'b',label='#nofilter')
plt.plot(chan_freqs,filtermags,'g',label='Filtered')
plt.legend()

plt.figure(2)
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
plt.plot(chan_freqs,mags,'g')
plt.plot(chan_freqs[ind_kids],mags[ind_kids],'r*')

# list of kid frequencies
target_freqs = chan_freqs[ind_kids]
