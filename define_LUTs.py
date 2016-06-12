import numpy as np

class roach_LUT(object):

	def __init__(self):
    		self.test_comb = np.linspace(-200.2340e6, 200.2450e6, 500)
        	self.dac_samp_freq = 512.0e6
        	self.fpga_samp_freq = 256.0e6
        	self.test_freq = np.array([50.0125]) * 1.0e6
        	self.LUTbuffer_len = 2**21
        	self.dac_freq_res = self.dac_samp_freq/self.LUTbuffer_len
        	self.fft_len = 1024
	
	def fft_bin_index(self, freqs, fft_len, samp_freq):
    	# returns the fft bin index for a given frequency, fft length, and sample frequency
        	return np.round((freqs/samp_freq)*fft_len).astype('int')
	
	def freq_comb(self, freqs, samp_freq, resolution,phase = np.array([0.]*1000), random_phase = True, DAC_LUT = True, amplitudes = np.array([1.]*1000)):
    	# Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q 
        	freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res
        	amp_full_scale = (2**15 - 1)
            	np.random.seed()
        	if DAC_LUT:
            		fft_len = self.LUTbuffer_len
            		bins = self.fft_bin_index(freqs, fft_len, samp_freq)
            		print bins
			phase = np.random.uniform(0., 2.*np.pi, len(bins))
			self.spec = np.zeros(fft_len,dtype='complex')
            		amps = np.array([1.]*len(bins))
            		self.spec[bins] = amps*np.exp(1j*(phase))
            		wave = np.fft.ifft(self.spec)
            		waveMax = np.max(np.abs(wave))
            		print waveMax
			I = (wave.real/waveMax)*(amp_full_scale)
            		Q = (wave.imag/waveMax)*(amp_full_scale)
        	else:
            		fft_len = (self.LUTbuffer_len/self.fft_len)
            		bins = self.fft_bin_index(freqs, fft_len, samp_freq)
            		print bins
			self.dds_spec = np.zeros(fft_len,dtype='complex')
			amps = np.array([1.]*len(bins))
            		self.dds_spec[bins] = amps*np.exp(1j*(phase))
            		wave = np.fft.ifft(self.dds_spec)
            		waveMax = np.max(np.abs(wave))
            		print waveMax
			I = (wave.real/waveMax)*(amp_full_scale)
            		Q = (wave.imag/waveMax)*(amp_full_scale)
		return I, Q    

	def select_bins(self, freqs):
        	bins = self.fft_bin_index(freqs, self.fft_len, self.dac_samp_freq)
        	bin_freqs = bins*self.dac_samp_freq/self.fft_len
        	self.freq_residuals = freqs - bin_freqs
		return

	def define_DDS_LUT(self,freqs):
		self.select_bins(freqs)
		I_dds, Q_dds = np.array([0.]*(self.LUTbuffer_len)), np.array([0.]*(self.LUTbuffer_len))
		for m in range(len(self.freq_residuals)):
			I, Q = self.freq_comb(np.array([self.freq_residuals[m]]), self.fpga_samp_freq/(self.fft_len/2.), self.dac_freq_res, random_phase = False, DAC_LUT = False)
			I_dds[m::1024] = I
		    	Q_dds[m::1024] = Q
        	return I_dds, Q_dds

	def make_luts(self, freqs):
		self.I_dac, self.Q_dac = self.freq_comb(freqs, self.dac_samp_freq, self.dac_freq_res, amplitudes = np.array([1.]*1000), random_phase = True)
        	self.I_dds, self.Q_dds = self.define_DDS_LUT(freqs)
		return

    	def main(self):
		self.make_luts(self.test_freq)
		return 

if __name__=='__main__':
    LUT = roach_LUT()
    LUT.main()
