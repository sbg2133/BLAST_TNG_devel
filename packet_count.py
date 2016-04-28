import matplotlib, time, struct
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import casperfpga 
import corr
from myQdr import Qdr as myQdr
import types
import logging
import glob  
import os
import sys
import valon_synth_glenn
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
from socket import *
from scipy import signal

class roachInterface(object):
	
	def __init__(self):
		self.v1 = valon_synth_glenn.Synthesizer('/dev/ttyUSB4')
		self.v1.set_frequency(0,512,0.01) # DAC/ADC
		self.dac_samp_freq = 512.0e6
		self.fpga_samp_freq = 256.0e6
		self.dds_shift = 304 # This varies b/t fpg/bof files
		self.port = 3000
		self.ip = '192.168.40.89' # Set to PPC IP in /etc/network/interfaces
		self.fpga = casperfpga.katcp_fpga.KatcpFpga(self.ip,timeout=120.)
		self.test_freq = np.array([50.0125]) * 1.0e6
		#self.freqs = np.array(np.loadtxt('BLASTResonatorPositionsVer2.txt', delimiter=','))
		self.freqs = np.array([152., 200.])*1.0e6 
		self.v1.set_frequency(8,512.0, 0.01) # LO
		self.LUTbuffer_len = 2**21
		self.dac_freq_res = self.dac_samp_freq/self.LUTbuffer_len
		self.f_base = 300.0
		self.fft_len = 1024
		self.test_bin = self.fft_bin_index(self.test_freq, self.fft_len, self.dac_samp_freq)
		#self.bb_freqs, delta_f = np.linspace(-200.523e6, 200.2345e6, 101,retstep=True)
		#self.fft_bins = self.fft_bin_index(self.bb_freqs, self.fft_len, self.dac_samp_freq)
		#self.test_bin = self.fft_bin_index(self.test_freq, self.fft_len, self.dac_samp_freq)
		self.main_prompt = '\n\t\033[35mROACHII mKID Readout\033[0m\n\t\033[33mChoose a number from the list and press Enter. 0 - 4 should be followed in order:\033[0m'
		self.main_opts= ['Calibrate QDR','Initialize GbE (Must toggle before writing first tone)','Write Test Tone','Write DAC, DDS LUTs','Stream UDP packets','VNA sweep and plot','Locate resonances','Target sweep and plot','Bind Socket', 'Exit'] 
		"""	
		self.UDP_IP = "192.168.41.2" # local ip of host pc
		self.dest_ip = 192*(2**24) + 168*(2**16) + 41*(2**8) + 2 
		self.dest_port = 60000
		self.fpga.write_int('tx_destip',self.dest_ip)
		self.fpga.write_int('tx_destport',self.dest_port)
		self.fpga.write_int('rx_ack', 1)
		self.fpga.write_int('rx_rst', 0)
		"""
		#self.UDP_IP = "192.168.41.2" 
		#self.UDP_PORT = 60000 # Fabric Port
		self.dest_ip  = 192*(2**24) + 168*(2**16) + 41*(2**8) + 2 # Set to FPGA IP in /etc/network/interfaces
		self.fabric_port= 60000 
		self.s = socket(AF_PACKET, SOCK_RAW, htons(3))
		self.s.setsockopt(SOL_SOCKET, SO_RCVBUF, 8192 + 42)
		self.s.bind(('eth0', 3))
		self.fpga.write_int('tx_destip',self.dest_ip)
		self.fpga.write_int('tx_destport',self.fabric_port)
		self.fpga.write_int('rx_ack', 1)
		
		self.accum_len = (2**19)-1 
		self.fpga.write_int('sync_accum_len', self.accum_len)
		self.accum_freq = self.fpga_samp_freq / self.accum_len # FPGA clock freq / accumulation length	
		self.fpga.write_int('fft_shift', 255)	
		self.fpga.write_int('dds_shift', self.dds_shift)
		self.save_path = '/mnt/iqstream/'

	def upload_fpg(self):
		print 'Connecting...'
		t1 = time.time()
		timeout = 10
		while not self.fpga.is_connected():
		    	if (time.time()-t1) > timeout:
				raise Exception("Connection timeout to roach")
		time.sleep(0.1)
		if (self.fpga.is_connected() == True):
			print 'Connection established to', self.ip
		    	self.fpga.upload_to_ram_and_program(str(self.bitstream))
		else:
		    	print 'Not connected to the FPGA'
		time.sleep(2)
		print 'Uploaded', self.bitstream
		return

	def qdrCal(self):	
	# Calibrates the QDRs. Run after writing to QDR.  	
		self.fpga.write_int('dac_reset',1)
		bQdrCal = True
		bQdrCal2 = True
		bFailHard = False
		calVerbosity = 1
		qdrMemName = 'qdr0_memory'
		qdrNames = ['qdr0_memory','qdr1_memory']
		print 'Fpga Clock Rate =',self.fpga.estimate_fpga_clock()
		if bQdrCal:
			self.fpga.get_system_information()
			results = {}
			for qdr in self.fpga.qdrs:
				print qdr
				if bQdrCal2:
					mqdr = myQdr.from_qdr(qdr)
					results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
				else:
					results[qdr.name] = qdr.qdr_cal(fail_hard=bFailHard,verbosity=calVerbosity)
			print 'qdr cal results:',results
			for qdrName in ['qdr0','qdr1']:
				if not results[qdr.name]:
					print 'Calibration Failed'
					break

	def toggle_dac(self):
		self.fpga.write_int('dac_reset',1)
		self.fpga.write_int('dac_reset',0)
		return
	
	def fft_bin_index(self, freqs, fft_len, samp_freq):
	# returns the fft bin index for a given frequency, fft length, and sample frequency
		bin_index = np.round((freqs/samp_freq)*fft_len).astype('int')
		return bin_index

	def read_mixer_snaps(self, shift, chan, mixer_out = True):
	# returns snap data for the dds mixer inputs and outputs
		self.fpga.write_int('dds_shift', shift)
		if (chan % 2) > 0: # if chan is odd
			self.fpga.write_int('chan_select', (chan - 1) / 2)
		else:
			self.fpga.write_int('chan_select', chan/2)
		self.fpga.write_int('rawfftbin_ctrl', 0)
		self.fpga.write_int('mixerout_ctrl', 0)
		self.fpga.write_int('rawfftbin_ctrl', 1)
		self.fpga.write_int('mixerout_ctrl', 1)
		mixer_in = np.fromstring(self.fpga.read('rawfftbin_bram', 16*2**14),dtype='>i2').astype('float')
		mixer_in /= 2.0**15
		if mixer_out:
			mixer_out = np.fromstring(self.fpga.read('mixerout_bram', 8*2**14),dtype='>i2').astype('float')
			mixer_out /= 2.0**14
			return mixer_in, mixer_out
		else:
			return mixer_in

	def return_shift(self, chan):
	# Returns the dds shift
		dds_spec = np.abs(np.fft.rfft(self.I_dds[chan::1024],1024))
		dds_index = np.where(np.abs(dds_spec) == np.max(np.abs(dds_spec)))[0][0]
		print 'Finding LUT shift...' 
		for i in range(512):
			print i
			mixer_in = self.read_mixer_snaps(i, chan, mixer_out = False)
			I0_dds_in = mixer_in[2::8]	
			I0_dds_in[np.where(I0_dds_in > 32767.)] -= 65535.
			snap_spec = np.abs(np.fft.rfft(I0_dds_in,1024))
			snap_index = np.where(np.abs(snap_spec) == np.max(np.abs(snap_spec)))[0][0]
			if dds_index == snap_index:
				print 'LUT shift =', i
				shift = i
				break
		return shift

	def mixer_comp(self,chan, find_shift = True, I0 = True, plot = True):
	# Plots the dds mixer data at the shift found by return_shift 	
		if find_shift:
			shift = self.return_shift(chan)
		else: 
			shift = self.dds_shift
			#shift = input('Shift = ?')
		mixer_in, mixer_out = self.read_mixer_snaps(shift, chan)	
		if I0:
			I_in = mixer_in[0::8]
			Q_in = mixer_in[1::8]
			I_dds_in = mixer_in[2::8]
			Q_dds_in = mixer_in[3::8]
			I_out = mixer_out[0::4]
			Q_out = mixer_out[1::4]
		else:
			I_in = mixer_in[4::8]
			Q_in = mixer_in[5::8]
			I_dds_in = mixer_in[6::8]
			Q_dds_in = mixer_in[7::8]
			I_out = mixer_out[2::4]
			Q_out = mixer_out[3::4]
		# Mixer in 
		I_out_guess = ((I_in * I_dds_in) + (Q_in * Q_dds_in))
		Q_out_guess = (-1.*(I_in * Q_dds_in) + (Q_in * I_dds_in))
		# Mixer out 
		if plot:
			plt.figure()
			if I0:
				plt.suptitle('DDS Shift = ' + str(shift) + ', Freq = ' + str(self.test_freq/1.0e6) + ' MHz,' + ' I0')
			else:
				plt.suptitle('DDS Shift = ' + str(shift) + ', Freq = ' + str(self.test_freq/1.0e6) + ' MHz,' + ' I1')
			plt.subplot(2,3,1)
			plt.plot(I_in, label = 'I in', color = 'black', linewidth = 2)
			plt.plot(I_dds_in, label = 'I dds in', color = 'red')
			plt.xlim((0,300))
			plt.ylim((-1.0,1.0))
			plt.legend()
			plt.grid()
			plt.subplot(2,3,2)
			plt.legend()
			plt.grid()
			plt.plot(Q_in, label = 'Q in', color = 'green', linewidth = 2)
			plt.plot(Q_dds_in, label = 'Q dds in', color = 'blue')
			plt.xlim((0,300))
			plt.ylim((-1.0,1.0))
			plt.legend()
			plt.grid()
			plt.subplot(2,3,3)
			plt.plot(I_dds_in, label = 'I dds in', color = 'red')
			plt.plot(Q_dds_in, label = 'Q dds in', color = 'blue')
			plt.xlim((0,300))
			plt.ylim((-1.0,1.0))
			plt.legend()
			plt.grid()
			plt.subplot(2,3,4)
			plt.plot(I_in, label = 'I in', color = 'black', linewidth = 2)
			plt.plot(Q_in, label = 'Q in', color = 'green', linewidth = 2)
			plt.xlim((0,300))
			plt.ylim((-1.0,1.0))
			plt.legend()
			plt.grid()
			plt.subplot(2,3,5)
			plt.plot(I_out_guess, label = 'I out predict', color = 'black', linewidth = 2)
			plt.plot(Q_out_guess, label = 'Q out predict', color = 'green', linewidth = 2)
			plt.xlim((0,300))
			plt.ylim((-2.0,2.0))
			plt.legend()
			plt.grid()
			plt.subplot(2,3,6)
			plt.plot(I_out, label = 'I out', color = 'black', linewidth = 2)
			plt.plot(Q_out, label = 'Q out', color = 'green', linewidth = 2)
			plt.xlim((0,300))
			plt.ylim((-2.0,2.0))
			plt.legend()
			plt.grid()
			plt.show()
		return I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out

	def plotMixer(self, chan):
		#chan = sys.argv[1]
		#chan = int(chan)
		fig = plt.figure(num= None, figsize=(12,12), dpi=80, facecolor='w', edgecolor='w')
		# I and Q
		plt.suptitle('Channel ' + str(chan) + ' , Tone freq = ' + str(self.test_freq[0]/1.0e6) + ' MHz',size = 20) 
		plot1 = fig.add_subplot(311)
		plt.title('I/Q into mixer', size = 20)
		line1, = plot1.plot(range(16384), np.zeros(16384), label = 'I in', color = 'green', linewidth = 2)
		line2, = plot1.plot(range(16384), np.zeros(16384), label = 'Q in', color = 'black', linewidth = 2)
		plt.xlim((0,500))
		plt.ylim((-1.0,1.0))
		plt.grid()
		# DDS I and Q
		plot2 = fig.add_subplot(312)
		plt.title('I/Q DDS into mixer', size = 20)
		line3, = plot2.plot(range(16384), np.zeros(16384), label = 'I dds', color = 'red', linewidth = 2)
		line4, = plot2.plot(range(16384), np.zeros(16384), label = 'Q dds', color = 'black', linewidth = 2)
		plt.xlim((0,500))
		plt.ylim((-1.0,1.0))
		plt.grid()
		# Mixer output
		plot3 = fig.add_subplot(313)
		plt.title('I/Q mixer out', size = 20)
		line5, = plot3.plot(range(16384), np.zeros(16384), label = 'I out', color = 'green', linewidth = 2)
		line6, = plot3.plot(range(16384), np.zeros(16384), label = 'Q out', color = 'black', linewidth = 2)
		plt.xlim((0,500))
		plt.ylim((-2.0, 2.0))
		plt.grid()
		plt.show(block = False)
		count = 0
		stop = 10000
		while (count < stop):
			if (chan % 2) > 0:
				I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out = self.mixer_comp(chan, find_shift = False, I0 = False, plot = False)
			else:
				I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out = self.mixer_comp(chan, find_shift = False, plot = False)
			line1.set_ydata(I_in)
			line2.set_ydata(Q_in)
			line3.set_ydata(I_dds_in)
			line4.set_ydata(Q_dds_in)
			line5.set_ydata(I_out)
			line6.set_ydata(Q_out)
			plt.draw()
			#plt.savefig('/home/user1/blastfirmware/images/' + 'mixer0' + str(int(count)) + '.png', dpi=fig.dpi)
			count += 1

	def freq_comb(self, freqs, samp_freq, resolution,phase = np.array([0.]*1000), random_phase = True, DAC_LUT = True, amplitudes = np.array([1.]*1000)):
	# Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q 
		freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res
		if DAC_LUT:
			fft_len = self.LUTbuffer_len
			bins = self.fft_bin_index(freqs, fft_len, samp_freq)
			amps = np.array([0.001]*len(bins))
			#amps[0] = 1.0e-2
		else:
			fft_len = (self.LUTbuffer_len/self.fft_len)
			bins = self.fft_bin_index(freqs, fft_len, samp_freq)
			amps = np.array([1.]*len(freqs))
		amp_full_scale = (2**15 - 1)
		spec = np.zeros(fft_len,dtype='complex')
		if random_phase:
			np.random.seed()
			phase = np.random.uniform(0., 2.*np.pi, len(bins))
		spec[bins] = amps*np.exp(1j*(phase))
		wave = np.fft.ifft(spec)
		waveMax = np.max(np.abs(wave))
		#I = wave.real * amp_full_scale
		#Q = wave.imag * amp_full_scale
		I = (wave.real/waveMax)*(amp_full_scale)
		Q = (wave.imag/waveMax)*(amp_full_scale)
		return I, Q	
	
	def calculate_amps_orig(self):
		sweep_freqs, Is, Qs = self.open_stored(save_path = np.load('/mnt/iqstream/last_vna_dir.npy')[0])
		channels = np.load('/mnt/iqstream/last_channels.npy')
		chan_mags = [10.*np.log10(np.sqrt(Is[:,chan]**2+Qs[:,chan]**2)) for chan in range(len(channels))]
		chan_avgs = [np.mean(chan_mags[chan]) for chan in range(len(channels))]
		offsets = chan_avgs - np.mean(chan_mags)
		amps = np.sqrt(10**(offsets/10.))
		return chan_mags, chan_avgs, offsets, amps	
	
	def calculate_amps(self):
		lo_freqs, Is, Qs = self.open_stored(save_path = np.load('/mnt/iqstream/last_vna_dir.npy')[0])
		channels = np.load('/mnt/iqstream/last_channels.npy')
		chan_mags = [10.*np.log10(np.sqrt(Is[:,chan][2:]**2+Qs[:,chan][2:]**2)) for chan in range(len(channels))]
		chan_avgs = [np.mean(chan_mags[chan]) for chan in range(len(channels))]
		offsets = chan_avgs - np.mean(np.delete(chan_mags,  [(len(chan_mags)/2) - 1,(len(chan_mags)/2), (len(chan_mags)/2) + 1], 0))
		amps = np.sqrt(10**(offsets/10.))
		amps = 1 - amps + 1
		return chan_mags, chan_avgs, offsets, amps	
	
	def calc_plot(self):
		chan_mags, chan_avgs, offsets, amps = self.calculate_amps()
		bb_freqs = np.load('/mnt/iqstream/last_bb_freqs.npy')
		channels = np.load('/mnt/iqstream/last_channels.npy')
		lo_freqs, Is, Qs = self.open_stored(save_path = np.load('/mnt/iqstream/last_vna_dir.npy')[0])
		#[plt.plot((lo_freqs[2:] + bb_freqs[chan])/1.0e9,10*np.log10(np.sqrt(Is[:,chan][2:]**2+Qs[:,chan][2:]**2))) for chan in range(len(channels))]
		for chan in range(len(channels)):
			if offsets[chan] < 0:
				high = np.mean(chan_mags)
				low = chan_avgs[chan]
			else:
				high = chan_avgs[chan]
				low = np.mean(chan_mags)
			
			plt.plot((lo_freqs[2:] + bb_freqs[chan])/1.0e9,10*np.log10(np.sqrt(Is[:,chan][2:]**2+Qs[:,chan][2:]**2))) 
			plt.plot(((750.0e6 + bb_freqs[chan])/1.0e9, (750.0e6 + bb_freqs[chan])/1.0e9), (low, high), 'k-')
		plt.axhline(np.mean(chan_mags), color = 'red')
		plt.show()
		return
	
	def select_bins(self, freqs):
	# Adjusts the DAC frequencies to the DAC frequency resolution and calculates the offset from each bin center, to be used as the DDS LUT frequencies
		bins = self.fft_bin_index(freqs, self.fft_len, self.dac_samp_freq)
		#print 'Bin numbers = ', bins
		bin_freqs = bins*self.dac_samp_freq/self.fft_len
		#print 'Bin center freqs = ', bin_freqs/1.0e6
		self.freq_residuals = np.round((freqs - bin_freqs)/self.dac_freq_res)*self.dac_freq_res
		ch = 0
		for fft_bin in bins:
			self.fpga.write_int('bins', fft_bin)#have fft_bin waiting at ram gate
		    	self.fpga.write_int('load_bins', 2*ch + 1)#enable write ram at address i
		    	self.fpga.write_int('load_bins', 0)#disable write 
		    	ch += 1
		# This is done to clear any unused channelizer RAM addresses
		for n in range(1024 - len(bins)):
			self.fpga.write_int('bins', 0)#have fft_bin waiting at ram gate
		   	self.fpga.write_int('load_bins', 2*ch + 1)#enable write ram at address i
		    	self.fpga.write_int('load_bins', 0)#disable write 
			ch += 1
			n += 1
		return 
	
	def define_DDS_LUT(self,freqs):
# Builds the DDS look-up-table from I and Q given by freq_comb. freq_comb is called with the sample rate equal to the sample rate for a single FFT bin. There are two bins returned for every fpga clock, so the bin sample rate is 256 MHz / half the fft length  
		self.select_bins(freqs)
		I_dds, Q_dds = np.array([0.]*(self.LUTbuffer_len)), np.array([0.]*(self.LUTbuffer_len))
		for m in range(len(self.freq_residuals)):
			I, Q = self.freq_comb(np.array([self.freq_residuals[m]]), self.fpga_samp_freq/(self.fft_len/2.), self.dac_freq_res, random_phase = False, DAC_LUT = False)
			I_dds[m::1024] = I
			Q_dds[m::1024] = Q
		return I_dds, Q_dds
	
	def pack_luts(self, freqs, amplitudes = np.array([1.]*1000)):
	# packs the I and Q look-up-tables into strings of 16-b integers, in preparation to write to the QDR. Returns the string-packed look-up-tables
		self.I_dac, self.Q_dac = self.freq_comb(freqs, self.dac_samp_freq, self.dac_freq_res, amplitudes)
		self.I_dds, self.Q_dds = self.define_DDS_LUT(freqs)
		self.I_lut, self.Q_lut = np.zeros(self.LUTbuffer_len*2), np.zeros(self.LUTbuffer_len*2)
		self.I_lut[0::4] = self.I_dac[1::2] 		
		self.I_lut[1::4] = self.I_dac[0::2]
		self.I_lut[2::4] = self.I_dds[1::2]
		self.I_lut[3::4] = self.I_dds[0::2]
		self.Q_lut[0::4] = self.Q_dac[1::2] 		
		self.Q_lut[1::4] = self.Q_dac[0::2]
		self.Q_lut[2::4] = self.Q_dds[1::2]
		self.Q_lut[3::4] = self.Q_dds[0::2]
		print 'String Packing LUT...',
		self.I_lut_packed = self.I_lut.astype('>i2').tostring()
		self.Q_lut_packed = self.Q_lut.astype('>i2').tostring()
		print 'Done.'
		return 
		
	def writeQDR(self, freqs, amplitudes = np.array([1.]*1000)):
	# Writes packed LUTs to QDR
		self.pack_luts(freqs, amplitudes)
		self.fpga.write_int('dac_reset',1)
		self.fpga.write_int('dac_reset',0)
		print 'Writing DAC and DDS LUTs to QDR...',
		self.fpga.write_int('start_dac',0)
		self.fpga.blindwrite('qdr0_memory',self.I_lut_packed,0)
		self.fpga.blindwrite('qdr1_memory',self.Q_lut_packed,0)
		self.fpga.write_int('start_dac',1)
		print 'Done.'
		return 

	def read_QDR_katcp(self):
	# Reads out QDR buffers with KATCP, as 16-b signed integers.	
		self.QDR0 = np.fromstring(self.fpga.read('qdr0_memory', 8 * 2**20),dtype='>i2')
		self.QDR1 = np.fromstring(self.fpga.read('qdr1_memory', 8* 2**20),dtype='>i2')
		self.I_katcp = self.QDR0.reshape(len(self.QDR0)/4.,4.)
		self.Q_katcp = self.QDR1.reshape(len(self.QDR1)/4.,4.)
		self.I_dac_katcp = np.hstack(zip(self.I_katcp[:,1],self.I_katcp[:,0]))
		self.Q_dac_katcp = np.hstack(zip(self.Q_katcp[:,1],self.Q_katcp[:,0]))
		self.I_dds_katcp = np.hstack(zip(self.I_katcp[:,3],self.I_katcp[:,2]))
		self.Q_dds_katcp = np.hstack(zip(self.Q_katcp[:,3],self.Q_katcp[:,2]))
		return		

	def read_QDR_snap(self):
	# Reads out QDR snaps
		self.fpga.write_int('QDR_LUT_snap_qdr_ctrl',0)
		self.fpga.write_int('QDR_LUT_snap_qdr_ctrl',1)
		qdr_snap = np.fromstring(self.fpga.read('QDR_LUT_snap_qdr_bram', 16 * 2**10),dtype='>i2').astype('float')
		self.QDRs = qdr_snap.reshape(len(qdr_snap)/8.,8.)
		self.I1_dds_snap = self.QDRs[:,0]
		self.I0_dds_snap = self.QDRs[:,1]
		self.I1_snap = self.QDRs[:,2]
		self.I0_snap = self.QDRs[:,3]
		self.Q1_dds_snap = self.QDRs[:,4]
		self.Q0_dds_snap = self.QDRs[:,5]
		self.Q1_snap = self.QDRs[:,6]
		self.Q0_snap = self.QDRs[:,7]
		self.I_dac_snap = np.hstack(zip(self.I0_snap,self.I1_snap))
		self.Q_dac_snap = np.hstack(zip(self.Q0_snap,self.Q1_snap))
		self.I_dds_snap = np.hstack(zip(self.I0_dds_snap,self.I1_dds_snap))
		self.Q_dds_snap = np.hstack(zip(self.Q0_dds_snap,self.Q1_dds_snap))
		return

	def read_chan_snaps(self):
	# Reads the snap blocks at the bin select RAM and channelizer mux
		self.fpga.write_int('buffer_out_ctrl', 0)
		self.fpga.write_int('buffer_out_ctrl', 1)
		self.chan_data = np.fromstring(ri.fpga.read('buffer_out_bram', 8 * 2**9),dtype = '>H')
		self.fpga.write_int('chan_bins_ctrl', 0)
		self.fpga.write_int('chan_bins_ctrl', 1)
		self.chan_bins = np.fromstring(ri.fpga.read('chan_bins_bram', 4 * 2**14),dtype = '>H')
		return

	def read_accum_snap(self):
        # Reads the avgIQ buffer. Returns I and Q as 32-b signed integers 	
		self.fpga.write_int('accum_snap_ctrl', 0)
        	self.fpga.write_int('accum_snap_ctrl', 1)
        	accum_data = np.fromstring(self.fpga.read('accum_snap_bram', 16*2**9), dtype = '>i').astype('float')
		accum_data /= 2.0**17
		accum_data /= ((self.accum_len)/512.)
		I0 = accum_data[0::4]	
		Q0 = accum_data[1::4]	
		I1 = accum_data[2::4]	
		Q1 = accum_data[3::4]	
		I = np.hstack(zip(I0, I1))
		Q = np.hstack(zip(Q0, Q1))
		return I, Q	

	def plotAccum(self):
	# Generates a plot stream from read_avgIQ_snap(). To view, run plotAvgIQ.py in a separate terminal
		fig = plt.figure(num= None, figsize=(12,12), dpi=80, facecolor='w', edgecolor='w')
		plt.suptitle('Averaged FFT, Accum. Frequency = ' + str(self.accum_freq), fontsize=20)
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(0,1024),np.zeros(1024), 'b')
		plt.xlabel('Channel #',fontsize = 20)
		plt.ylabel('Amplitude',fontsize = 20)
		plt.xticks(np.arange(0,1024,100))
		plt.xlim(-50,1075)
		plt.grid()
		plt.show(block = False)
		count = 0 
		stop = 10000
		while(count < stop):
			I, Q = self.read_accum_snap()
			mags = np.sqrt(I**2 + Q**2)
			plt.ylim((0,np.max(mags) + 0.001))
			line1.set_ydata(mags)
			plt.draw()
			count += 1
		#	plt.savefig('/home/user1/blastfirmware/images/' + 'accum' + str(int(count)) + '.png', dpi=fig.dpi)
		return

	def plotADC(self):
	# Plots the ADC timestream
	# Peak to peak should be 900 mV (from DAC)
		fig = plt.figure(num= None, figsize=(12,12), dpi=80, facecolor='w', edgecolor='w')
		ax = fig.add_subplot(111)	
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		#ax.suptitle('ADC timestream', size = 20)
		ax.set_xlabel('ADC Clock Period (1/512 MHz =  2 ns)', size = 20)
		ax.set_ylabel('Amplitude (Volts)', size = 20)
		plot1 = fig.add_subplot(211)
		line1, = plot1.plot(np.arange(0,2048), np.zeros(2048), 'r-', linewidth = 2)
		plot1.set_title('I', size = 20)
		plt.xlim(0,100)
		plt.ylim(-1.1,1.1)
		#plt.yticks(np.arange(-4e4, 4e4, 5000.))
		plt.grid()
		plot2 = fig.add_subplot(212)
		line2, = plot2.plot(np.arange(0,2048), np.zeros(2048), 'b-', linewidth = 2)
		plot2.set_title('Q', size = 20)
		plt.xlim(0,100)
		plt.ylim(-1.1,1.1)
		#plt.yticks(np.arange(-4e4, 4e4, 5000.))
		plt.grid()
		plt.show(block = False)
		count = 0
		stop = 1.0e6
		while count < stop:	
			time.sleep(0.1)
			self.fpga.write_int('adc_snap_ctrl',0)
			self.fpga.write_int('adc_snap_ctrl',1)
			self.fpga.write_int('adc_snap_trig',0)    
			self.fpga.write_int('adc_snap_trig',1)    
			self.fpga.write_int('adc_snap_trig',0)
			adc = (np.fromstring(self.fpga.read('adc_snap_bram',(2**10)*8),dtype='>i2')).astype('float')
			adc /= 2.0**15 
			# ADC full scale is 2.2 V
			#adc *= 0.909091
			I = np.hstack(zip(adc[0::4],adc[1::4]))
			Q = np.hstack(zip(adc[2::4],adc[3::4]))
			#return I
			#raw_input()
			line1.set_ydata(I)
			line2.set_ydata(Q)
			plt.draw()
			count += 1
			#plt.savefig('/home/user1/blastfirmware/images/' + 'adc' + str(int(count)) + '.png', dpi=fig.dpi)
	
	def plotFFT(self):
	# Generates plot of the FFT output. To view, run plotFFT.py in a separate terminal
		fig = plt.figure(num= None, figsize=(12,12), dpi=80, facecolor='w', edgecolor='w')
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot( np.arange(0,512,0.5), np.zeros(1024), 'b-')
		plt.xlabel('freq (MHz)',fontsize = 20)
		plt.ylabel('mV',fontsize = 20)
		plt.title('Pre-mixer FFT',fontsize = 20)
		plt.xticks(np.arange(0,512,50))
		plt.xlim((0,512))
		plt.grid()
		plt.show(block = False)
		count = 0 
		stop = 1.0e6
		while(count < stop):
			overflow = np.fromstring(self.fpga.read('overflow', 4), dtype = '>B')
			print overflow
			self.fpga.write_int('fft_snap_ctrl',0)
			self.fpga.write_int('fft_snap_ctrl',1)
			fft_snap = (np.fromstring(self.fpga.read('fft_snap_bram',(2**9)*8),dtype='>i2')).astype('float')
			I0 = fft_snap[0::4]
			Q0 = fft_snap[1::4]
			I1 = fft_snap[2::4]
			Q1 = fft_snap[3::4]
			mag0 = np.sqrt(I0**2 + Q0**2)
			mag1 = np.sqrt(I1**2 + Q1**2)
			fft_mags = np.hstack(zip(mag0,mag1))
			fft_mags /= 2.0**17
			fft_mags *= 1000. # put into mV
			#fft_mags = 10.**(fft_mags/10.)
			plt.ylim((0,np.max(fft_mags)))
			line1.set_ydata((fft_mags))
			plt.draw()
			#plt.savefig('/home/user1/blastfirmware/images/' + 'fft' + str(int(count)) + '.png', dpi=fig.dpi)
			count += 1
	
	def plotPhase(self, chan):
		#chan = sys.argv[1]
		chan = int(chan) + 2
		count = 0 
		stop = 1.0e6
		while(count < stop):
			time.sleep(0.1)
			I, Q = self.read_accum_snap()
			phase = np.arctan2(Q[chan],I[chan])
			#phase = np.rad2deg(phase)
			print 'Phase =', np.round(phase,10), I[chan], Q[chan]
			count += 1
		return 

	def initialize_GbE(self):
		# Configure GbE Block. Run immediately after calibrating QDR.
		#self.s1 = socket(AF_INET,SOCK_DGRAM) # socket for roach board 1
		#self.s1.bind((self.UDP_IP,self.dest_port))
		self.fpga.write_int('tx_rst',0)
		self.fpga.write_int('tx_rst',1)
		self.fpga.write_int('tx_rst',0)
		return

	def bind_socket(self):
		self.dest_port = input('Port number ? ')
		self.fpga.write_int('tx_destport',self.dest_port)
		self.s1 = socket(AF_INET,SOCK_DGRAM) # socket for roach board 1
		self.s1.bind((self.UDP_IP,self.dest_port))
		return	

	def stream_UDP(self, chan, Npackets):
		self.fpga.write_int('pps_start', 1)
		#self.phases = np.empty((len(self.freqs),Npackets))
		phases = np.empty(Npackets)
		count = 0
		while count < Npackets:
			packet = self.s.recv(8234) # total number of bytes including 42 byte header
			#header = np.fromstring(packet[:42],dtype = '<B')
			#roach_mac = header[6:12]
			#filter_on = np.array([2, 68, 1, 2, 13, 33])
			#if np.array_equal(roach_mac,filter_on):
			data = np.fromstring(packet[42:],dtype = '<i').astype('float')
			#print data
			#raw_input()
			#data = np.fromstring(packet,dtype = '<i').astype('float')
			data /= 2.0**17
			data /= (self.accum_len/512.)
			ts = (np.fromstring(packet[-5:-1],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
			packet_count = (np.fromstring(packet[-1],dtype = 'B')).astype('float')
			# To stream one channel, make chan an argument
			if (chan % 2) > 0:
				I = data[1024 + ((chan - 1) / 2)]	
				Q = data[1536 + ((chan - 1) /2)]	
			else:
				I = data[0 + (chan/2)]	
				Q = data[512 + (chan/2)]	
			phase = np.arctan2([Q],[I])
			#np.save('/mnt/iqstream/packet' + str(count) +'.npy',packet)
			"""
			odd_chan = self.channels[1::2]
			even_chan = self.channels[0::2]
			I_odd = data[1024 + ((odd_chan - 1) / 2)]	
			Q_odd = data[1536 + ((odd_chan - 1) /2)]	
			I_even = data[0 + (even_chan/2)]	
			Q_even = data[512 + (even_chan/2)]	
			even_phase = np.arctan2(Q_even,I_even)
			odd_phase = np.arctan2(Q_odd,I_odd)
			phase = np.hstack(zip(even_phase, odd_phase))
			self.phases[count] = phase
			"""
			phases[count]=phase
			print count,packet_count,ts,phase
			#else:
			#	continue
			count += 1
		return 
	
	def target_sweep(self, save_path = '/mnt/iqstream/target_sweeps', write = True, span = 100.0e3):
		write = raw_input('Write ? (y/n) ')
		kid_freqs = np.load('/mnt/iqstream/last_kid_freqs.npy')
		sweep_dir = raw_input('Target sweep dir ? ')
		save_path = os.path.join(save_path, sweep_dir)
		#kid_freqs = np.array(np.loadtxt('BLASTResonatorPositionsVer2.txt', delimiter=','))
		center_freq = (np.max(kid_freqs) + np.min(kid_freqs))/2.   #Determine LO position to put tones centered around LO
		self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
		bb_freqs = kid_freqs - center_freq
		bb_freqs = np.roll(bb_freqs, - np.argmin(np.abs(bb_freqs)) - 1)
		np.save('/mnt/iqstream/last_bb_freqs.npy',bb_freqs)
		rf_freqs = bb_freqs + center_freq
		np.save('/mnt/iqstream/last_rf_freqs.npy',rf_freqs)
		channels = np.arange(len(rf_freqs))
		np.save('/mnt/iqstream/last_channels.npy',channels)
		self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
		print '\nTarget baseband freqs (MHz) =', bb_freqs/1.0e6
		print '\nTarget RF freqs (MHz) =', rf_freqs/1.0e6
		if write == 'y':\
			self.writeQDR(bb_freqs)
		self.fpga.write_int('sync_accum_reset', 0)
		self.fpga.write_int('sync_accum_reset', 1)
		self.sweep_lo(Npackets_per = 10, channels = channels, center_freq = center_freq, span = span , save_path = save_path)
		last_target_dir = save_path
		np.save('/mnt/iqstream/last_target_dir.npy',np.array([last_target_dir]))
		self.plot_kids(save_path = last_target_dir, bb_freqs = bb_freqs, channels = channels)
		#plt.figure()
		#plt.plot()
		return

	def vna_sweep(self, center_freq = 750.0e6, save_path = '/mnt/iqstream/vna_sweeps', write = True):
		write = raw_input('Write no cal? (y/n)')
		calc = raw_input('Calculate amplitudes? (y/n)')
		sweep_dir = raw_input('VNA sweep dir ? ')
		save_path = os.path.join(save_path, sweep_dir)
		bb_freqs, delta_f = np.linspace(-200.0e6, 200.0e6, 1000,retstep=True)
		bb_freqs = np.roll(bb_freqs, - np.argmin(np.abs(bb_freqs)) - 1)
		np.save('/mnt/iqstream/last_bb_freqs.npy',bb_freqs)
		rf_freqs = bb_freqs + center_freq
		np.save('/mnt/iqstream/last_rf_freqs.npy',rf_freqs)
		channels = np.arange(len(rf_freqs))
		self.calc_channels = channels
		np.save('/mnt/iqstream/last_channels.npy',channels)
		self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
		print '\nVNA baseband freqs (MHz) =', bb_freqs/1.0e6
		print '\nVNA RF freqs (MHz) =', rf_freqs/1.0e6
			#self.writeQDR(bb_freqs, amplitudes = amps)
		if write=='y':
			self.writeQDR(bb_freqs)
			self.fpga.write_int('sync_accum_reset', 0)
			self.fpga.write_int('sync_accum_reset', 1)
			self.sweep_lo(Npackets_per = 10, channels = channels, center_freq = center_freq, span = delta_f, save_path = save_path)
			last_vna_dir = save_path
                	np.save('/mnt/iqstream/last_vna_dir.npy',np.array([last_vna_dir]))
		if calc == 'y':
			chan_mags, chan_avgs, offsets, amps = self.calculate_amps()
			self.writeQDR(bb_freqs, amplitudes = amps)
			self.fpga.write_int('sync_accum_reset', 0)
			self.fpga.write_int('sync_accum_reset', 1)
			self.sweep_lo(Npackets_per = 10, channels = channels, center_freq = center_freq, span = delta_f, save_path = save_path)
			last_vna_dir = save_path
                	np.save('/mnt/iqstream/last_vna_dir.npy',np.array([last_vna_dir]))
			self.calc_plot()	
		else:
			self.plot_kids(save_path = last_vna_dir, bb_freqs = bb_freqs, channels = channels)
		return

	def sweep_lo(self, Npackets_per = 10, channels = None, center_freq = 750.0e6, span = 2.0e6, save_path = '/mnt/iqstream/lo_sweeps'):
		N = Npackets_per
		start = center_freq - (span/2.)
		stop = center_freq + (span/2.) 
		step = 100.0e3
		sweep_freqs = np.arange(start, stop, step)
		sweep_freqs = np.round(sweep_freqs/step)*step
		print 'Sweep freqs =', sweep_freqs/1.0e6
	 	if os.path.exists(save_path):
			[os.remove(os.path.join(save_path,fl)) for fl in os.listdir(save_path)]
		else:
			os.mkdir(save_path)
		for freq in sweep_freqs:
			print 'Sweep freq =', freq/1.0e6
			if self.v1.set_frequency(0, freq/1.0e6, 0.01): 
				time.sleep(0.1)
				self.store_UDP(N,freq, save_path,channels=channels) 
		self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
		return

	def store_UDP(self, Npackets, LO_freq, save_path, skip_packets=2, channels = None):
		#Npackets = np.int(time_interval * self.accum_freq)
		I_buffer = np.empty((Npackets + skip_packets, len(channels)))
		Q_buffer = np.empty((Npackets + skip_packets, len(channels)))
		self.fpga.write_int('pps_start', 1)
		count = 0
		while count < Npackets + skip_packets:
			packet = self.s1.recv(8234) # total number of bytes including 42 byte header
			#data = np.fromstring(packet,dtype = '<i').astype('float')
			data = np.fromstring(packet[42:],dtype = '<i').astype('float')
			data /= 2.0**17
			data /= (self.accum_len/512.)
			ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
			odd_chan = channels[1::2]
			even_chan = channels[0::2]
			I_odd = data[1024 + ((odd_chan - 1) / 2)]	
			Q_odd = data[1536 + ((odd_chan - 1) /2)]	
			I_even = data[0 + (even_chan/2)]	
			Q_even = data[512 + (even_chan/2)]	
			even_phase = np.arctan2(Q_even,I_even)
			odd_phase = np.arctan2(Q_odd,I_odd)
			if len(channels) % 2 > 0:
				I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
				Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
				I = np.hstack((I, I_even[-1]))	
				Q = np.hstack((Q, Q_even[-1]))	
				I_buffer[count] = I
				Q_buffer[count] = Q
			else:
				I = np.hstack(zip(I_even, I_odd))
				Q = np.hstack(zip(Q_even, Q_odd))
				I_buffer[count] = I
				Q_buffer[count] = Q
				
			count += 1
		I_file = 'I' + str(LO_freq)
		Q_file = 'Q' + str(LO_freq)
		np.save(os.path.join(save_path,I_file), np.mean(I_buffer[skip_packets:], axis = 0)) 
		np.save(os.path.join(save_path,Q_file), np.mean(Q_buffer[skip_packets:], axis = 0)) 
		return 

	
	def store_UDP_noavg(self, Npackets, LO_freq, save_path, skip_packets=2, channels = None):
		#Npackets = np.int(time_interval * self.accum_freq)
		I_buffer = np.empty((Npackets + skip_packets, len(channels)))
		Q_buffer = np.empty((Npackets + skip_packets, len(channels)))
		self.fpga.write_int('pps_start', 1)
		count = 0
		while count < Npackets + skip_packets:
			packet = self.s1.recv(8192 + 42) # total number of bytes including 42 byte header
			data = np.fromstring(packet[42:],dtype = '<i').astype('float')
			data /= 2.0**17
			data /= (self.accum_len/512.)
			ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
			odd_chan = channels[1::2]
			even_chan = channels[0::2]
			I_odd = data[1024 + ((odd_chan - 1) / 2)]	
			Q_odd = data[1536 + ((odd_chan - 1) /2)]	
			I_even = data[0 + (even_chan/2)]	
			Q_even = data[512 + (even_chan/2)]	
			even_phase = np.arctan2(Q_even,I_even)
			odd_phase = np.arctan2(Q_odd,I_odd)
			if len(channels) % 2 > 0:
				I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
				Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
				I = np.hstack((I, I_even[-1]))	
				Q = np.hstack((Q, Q_even[-1]))	
				I_buffer[count] = I
				Q_buffer[count] = Q
			else:
				I = np.hstack(zip(I_even, I_odd))
				Q = np.hstack(zip(Q_even, Q_odd))
				I_buffer[count] = I
				Q_buffer[count] = Q
				
			count += 1
		I_file = 'I' + str(LO_freq)
		Q_file = 'Q' + str(LO_freq)
		np.save(os.path.join(save_path,I_file), I_buffer[skip_packets:]) 
		np.save(os.path.join(save_path,Q_file), Q_buffer[skip_packets:]) 
		return 

	def open_stored(self, save_path = None):
		files = sorted(os.listdir(save_path))
		sweep_freqs = np.array([np.float(filename[1:-4]) for filename in files if (filename.startswith('I'))])
		I_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('I')]
		Q_list = [os.path.join(save_path, filename) for filename in files if filename.startswith('Q')]
		Is = np.array([np.load(filename) for filename in I_list])
		Qs = np.array([np.load(filename) for filename in Q_list])
		return sweep_freqs, Is, Qs

	def plot_kids(self, save_path = None, bb_freqs = None, channels = None):
		sweep_freqs, Is, Qs = self.open_stored(save_path)
		#[ plt.plot((sweep_freqs[2:] + bb_freqs[chan])/1.0e9,10*np.log10(np.sqrt(Is[:,chan][2:]**2+Qs[:,chan][2:]**2))) for chan in channels]
		scaled_mags = np.zeros((1000,205))
		for chan in range(np.shape(Is)[1]):
			mags = 10*np.log10(np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2))
			diff = 0 - np.mean(mags)
			scaled_mags[chan] = diff + mags
		
		[ plt.plot((sweep_freqs + bb_freqs[chan])/1.0e9,scaled_mags[chan]) for chan in range(np.shape(Is)[1])]
		plt.xlabel('Frequency (GHz)')
		plt.ylabel('10log(mag) [dBm]')
		#plt.savefig(os.path.join(save_path,'fig.png'))
		plt.show()
		return

        def lowpass(self,data,f0,order=1):
		size=data.size
		n=size
		#n=np.int(2.**(1.0+np.fix(np.log2(size))))
		df  = np.fft.rfft(data,n=n)
		df /= (1.0+np.power(np.arange(n/2+1)/np.float(n)/f0, 2.0*order))
		data = np.fft.irfft(df)
		return data

	def find_kids_vna(self,save_path=None):
		bb_freqs = np.load(os.path.join(self.save_path,'last_bb_freqs.npy')) 
		if save_path==None:
			save_path = np.load('/mnt/iqstream/last_vna_dir.npy')[0]	
		sweep_freqs, Is, Qs = self.open_stored(save_path = save_path)
		#concatenate and sort sweeps
		channels = np.load('/mnt/iqstream/last_channels.npy')
		Icat = np.concatenate([Is[:,chan] for chan in channels])
		Qcat = np.concatenate([Qs[:,chan] for chan in channels])
		freqs_cat = np.concatenate([sweep_freqs + bb_freqs[chan] for chan in channels])
		Icat = Icat[np.argsort(freqs_cat)]
		Qcat = Qcat[np.argsort(freqs_cat)]
		freqs_cat = freqs_cat[np.argsort(freqs_cat)]
		#phase slope:
		dphi = np.diff(np.unwrap(np.arctan2(Qcat,Icat)))
		#remove step spikes
		dphi[len(sweep_freqs)-1::len(sweep_freqs)]=dphi[len(sweep_freqs)::len(sweep_freqs)]
		plt.figure(figsize = (22,16))
		threshold_pos = 0.1
		threshold_neg = -1.
		plt.subplot(3,1,1)
		plt.plot(freqs_cat[1:],dphi)
		plt.xlim((450.0e6, 1050.0e6))
		plt.ylabel('rad/sample')
		plt.title(r'Raw d$\phi$ (rad)')
		#smooth data
		dphi = signal.convolve(dphi,signal.gaussian(100,3),mode='same')
		#find maxima
		startidx = np.where(np.diff((dphi>=threshold_pos).astype(int)) > 0)[0]
		stopidx  = np.where(np.diff((dphi>=threshold_pos).astype(int)) < 0)[0] + 1
		stopidx = np.append(stopidx,-1)
		kididx_pos  = np.array([i0 + np.argmax(dphi[i0:i1]) for i0,i1 in zip(startidx,stopidx)])

		startidx = np.where(np.diff((dphi<=threshold_neg).astype(int)) > 0)[0]
		stopidx  = np.where(np.diff((dphi<=threshold_neg).astype(int)) < 0)[0] + 1
		stopidx = np.append(stopidx,-1)
		kididx_neg  = np.array([i0 + np.argmin(dphi[i0:i1]) for i0,i1 in zip(startidx,stopidx)])
		print kididx_pos, kididx_neg
		kididx = np.sort(np.append(kididx_pos,kididx_neg))
		print kididx
		kid_freqs = (freqs_cat[1:]-(freqs_cat[1]-freqs_cat[0])/2.)[kididx]
		print 'Resonances at: ', kid_freqs/1.0e9
		print 'Found %d kids'%len(kid_freqs)
		print len(freqs_cat[1:]),len(dphi)
		plt.subplot(3,1,2)
		plt.plot(freqs_cat[1:],dphi)
		plt.plot(freqs_cat[1:][kididx],dphi[kididx],'ro')
		plt.hlines([threshold_pos,threshold_neg],freqs_cat.min(),freqs_cat.max())
		plt.ylabel('rad/sample')
		plt.xlim((450.0e6, 1050.0e6))
		plt.title('Smoothed phase grad')
		#plt.show()
		plt.subplot(3,1,3)
		plt.plot(freqs_cat,10*np.log10(np.sqrt(Icat**2+Qcat**2)))
		plt.plot(kid_freqs,10*np.log10(np.sqrt(Icat**2+Qcat**2))[kididx],'ro')
		plt.xlim((450.0e6, 1050.0e6))
		plt.xlabel('Frequency (GHz)')
		plt.ylabel('10log (S21 mag) [dB]')
		plt.title(r'250 $\mu$$m$ VNA sweep')
		plt.tight_layout()
		#plt.suptitle(r'BLAST-TNG 250$\mu$m array, ROACH2 sweep, # KIDS found = %d'%(len(self.kid_freqs)))
		plt.savefig(os.path.join(save_path,'fig.png'))
		plt.show()
		np.save('/mnt/iqstream/last_kid_freqs.npy',kid_freqs)
		return

	def get_stream(self, chan, time_interval):
		self.fpga.write_int('pps_start', 1)
		#self.phases = np.empty((len(self.freqs),Npackets))
		Npackets = np.int(time_interval * self.accum_freq)
		Is = np.empty(Npackets)
		Qs = np.empty(Npackets)
		phases = np.empty(Npackets)
		count = 0
		while count < Npackets:
			packet = self.s1.recv(8192 + 42) # total number of bytes including 42 byte header
			header = np.fromstring(packet[:42],dtype = '<B')
			roach_mac = header[6:12]
			filter_on = np.array([2, 68, 1, 2, 13, 33])
			if np.array_equal(roach_mac,filter_on):
				data = np.fromstring(packet[42:],dtype = '<i').astype('float')
				data /= 2.0**17
				data /= (self.accum_len/512.)
				ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
				# To stream one channel, make chan an argument
				if (chan % 2) > 0:
					I = data[1024 + ((chan - 1) / 2)]	
					Q = data[1536 + ((chan - 1) /2)]	
				else:
					I = data[0 + (chan/2)]	
					Q = data[512 + (chan/2)]	
				phase = np.arctan2([Q],[I])
				Is[count]=I
				Qs[count]=Q
				phases[count]=phase
			else:
				continue
			count += 1
		return Is, Qs, phases
	
	def plotPSD(self, chan, time_interval):
		Npackets = np.int(time_interval * self.accum_freq)
		plot_range = (Npackets / 2) + 1
		figure = plt.figure(num= None, figsize=(12,12), dpi=80, facecolor='w', edgecolor='w')
		# I 
		plt.suptitle('Channel ' + str(chan) + ' , Freq = ' + str((self.freqs[chan] + self.LO_freq)/1.0e6) + ' MHz') 
		plot1 = figure.add_subplot(311)
		plot1.set_xscale('log')
		plot1.set_autoscale_on(True)
		plt.ylim((-160,-80))
		plt.title('I')
		line1, = plot1.plot(np.linspace(0, self.accum_freq/2., (Npackets/2) + 1), np.zeros(plot_range), label = 'I', color = 'green', linewidth = 1)
		plt.grid()
		# Q
		plot2 = figure.add_subplot(312)
		plot2.set_xscale('log')
		plot2.set_autoscale_on(True)
		plt.ylim((-160,-80))
		plt.title('Q')
		line2, = plot2.plot(np.linspace(0, self.accum_freq/2., (Npackets/2) + 1), np.zeros(plot_range), label = 'Q', color = 'red', linewidth = 1)
		plt.grid()
		# Phase
		plot3 = figure.add_subplot(313)
		plot3.set_xscale('log')
		plot3.set_autoscale_on(True)
		plt.ylim((-120,-70))
		#plt.xlim((0.0001, self.accum_freq/2.))
		plt.title('Phase')
		plt.ylabel('dBc rad^2/Hz')
		plt.xlabel('log Hz')
		line3, = plot3.plot(np.linspace(0, self.accum_freq/2., (Npackets/2) + 1), np.zeros(plot_range), label = 'Phase', color = 'black', linewidth = 1)
		plt.grid()
		plt.show(block = False)
		count = 0
		stop = 1.0e10
		while count < stop:
			Is, Qs, phases = self.get_stream(chan, time_interval)
			I_mags = np.fft.rfft(Is, Npackets)
			Q_mags = np.fft.rfft(Is, Npackets)
			phase_mags = np.fft.rfft(phases, Npackets)
			I_vals = (np.abs(I_mags)**2 * ((1./self.accum_freq)**2 / (1.0*time_interval)))
			Q_vals = (np.abs(Q_mags)**2 * ((1./self.accum_freq)**2 / (1.0*time_interval)))
			phase_vals = (np.abs(phase_mags)**2 * ((1./self.accum_freq)**2 / (1.0*time_interval)))
			phase_vals = 10*np.log10(phase_vals)
			phase_vals -= phase_vals[0]
			#line1.set_ydata(Is)
			#line2.set_ydata(Qs)
			#line3.set_ydata(phases)
			line1.set_ydata(10*np.log10(I_vals))
			line2.set_ydata(10*np.log10(Q_vals))
			line3.set_ydata(phase_vals)
			plot1.relim()
			plot1.autoscale_view(True,True,False)
			plot2.relim()
			plot2.autoscale_view(True,True,False)
			#plot3.relim()
			plot3.autoscale_view(True,True,False)
			plt.draw()
			count +=1
		return

	def programLO(self, freq=800.0e6, sweep_freq=0):
		self.vi.simple_set_freq(8,freq)
		return

	def menu(self,prompt,options):
		print '\t' + prompt + '\n'
		for i in range(len(options)):
			print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
		opt = input()
		return opt
	
	def main_opt(self):
		while True:
			opt = self.menu(self.main_prompt,self.main_opts)
			if opt == 2:
				print '\nTest tone (MHz) =', self.test_freq/1e6
				self.writeQDR(self.test_freq)
				self.fpga.write_int('sync_accum_reset', 0)
				self.fpga.write_int('sync_accum_reset', 1)
			if opt == 0:
				os.system('clear')
				self.qdrCal()
			if opt == 3:
				print '\nDAC freqs (MHz) =', self.freqs/1e6
				print 'Length of Freq Comb =', len(self.freqs)
				self.writeQDR(self.freqs)
				self.fpga.write_int('sync_accum_reset', 0)
				self.fpga.write_int('sync_accum_reset', 1)
			if opt == 4:
				Npackets = input('\nNumber of UDP packets to stream? ' )
				chan = input('chan = ? ')
				self.stream_UDP(chan,Npackets)
			if opt == 1:
				self.initialize_GbE()
			if opt == 5:
				self.vna_sweep()
			if opt == 6:
				self.find_kids_vna()
			if opt == 7:
				self.target_sweep()
			if opt == 8:
				self.bind_socket()
			if opt == 9:
				sys.exit()

		return
	
	def main(self):
		os.system('clear')
		while True: 
			self.main_opt()

if __name__=='__main__':
	ri = roachInterface()
	ri.main()
