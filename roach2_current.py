# This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) May 23, 2016  Gordon, Sam <sbgordo1@asu.edu>
# Author: Gordon, Sam <sbgordo1@asu.edu>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
import valon_synth
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
from socket import *
from scipy import signal
import find_kids_blast 
import plot_sweep

class roachInterface(object):
    
    def __init__(self):
        #self.v1 = valon_synth.Synthesizer('/dev/ttyUSB4')
        self.ip = '192.168.40.52' # Set to PPC IP in /etc/network/interfaces
	self.fpga = casperfpga.katcp_fpga.KatcpFpga(self.ip,timeout=120.)
        #self.v1.set_frequency(0,750.0, 0.01)
        self.dds_shift = 305
	self.dac_samp_freq = 512.0e6
        self.fpga_samp_freq = 256.0e6
	#self.scale_factors[0], self.scale_factors[1] = 0., 0.
	#self.test_comb = np.array([50.0125])*1.0e6
        ##self.light_rf_freqs = np.load('/mnt/iqstream/lightfreqs.npy')
        #self.light_rf_freqs = np.load('/mnt/iqstream/sam_lightfreqs.npy')
        #self.center_freq = (np.max(self.light_rf_freqs) + np.min(self.light_rf_freqs))/2.
        #self.freqs = self.light_rf_freqs - 750.0e6
        #self.freqs = np.roll(self.freqs, - np.argmin(np.abs(self.freqs)) - 1)
        #self.test_comb = np.linspace(10.234234e6, 200.23e6, 500)
	neg_freqs, neg_delta = np.linspace(-255.021234e6+5.0e4, -5.2342e6+5.0e4, 150, retstep = True)
	#neg_freqs, neg_delta = np.linspace(-200., -20.2342e6 + 5.0e4, 200, retstep = True)
	pos_freqs, pos_delta = np.linspace(5.2342e6,255.021234e6, 150, retstep = True)
	#pos_freqs, pos_delta = np.linspace(20.2342e6,200.021234e6, 150, retstep = True)
	print neg_delta, pos_delta
	self.test_comb = np.concatenate((neg_freqs, pos_freqs))
	#self.test_comb = np.array([50.01234])*1.0e6
	#self.test_comb = np.linspace(-150.01213e6, 150.021234e6, 80)
	self.test_comb = self.test_comb[ self.test_comb != 0]
	self.test_comb = np.roll(self.test_comb, - np.argmin(np.abs(self.test_comb)) - 1)
	self.LUTbuffer_len = 2**21
        self.dac_freq_res = self.dac_samp_freq/self.LUTbuffer_len
        self.fft_len = 1024
        self.accum_len = (2**19) # 2**20 - 1 for 244 Hz 
        #self.s = socket(AF_PACKET, SOCK_RAW, htons(3))
        #self.s.setsockopt(SOL_SOCKET, SO_RCVBUF, 8192 + 42)
	#self.s.bind(('eth0', 3))
        self.main_prompt = '\n\t\033[33mKID-PY ROACH2 Readout\033[0m\n\t\033[35mChoose number and press Enter\033[0m'
        self.main_opts= ['Initialize','Write Test Comb','Stream UDP packets','VNA sweep and plot','Locate resonances','Target sweep and plot', 'Exit'] 

    def upload_fpg(self):
        print 'Connecting to ROACHII on',self.ip,'...'
        t1 = time.time()
        timeout = 10
        while not self.fpga.is_connected():
                if (time.time()-t1) > timeout:
                    raise Exception("Connection timeout to roach")
        time.sleep(0.1)
        if (self.fpga.is_connected() == True):
            print 'Connection established'
            self.fpga.upload_to_ram_and_program(self.bitstream)
        else:
                print 'Not connected to the FPGA'
        time.sleep(2)
        print 'Uploaded', self.bitstream
        return

    def qdrCal(self):    
    # Calibrates the QDRs. Run after writing to QDR.      
        self.fpga.write_int('dac_reset',1)
        print 'DAC on'
        bFailHard = False
        calVerbosity = 1
        qdrMemName = 'qdr0_memory'
        qdrNames = ['qdr0_memory','qdr1_memory']
        print 'Fpga Clock Rate =',self.fpga.estimate_fpga_clock()
        self.fpga.get_system_information()
        results = {}
        for qdr in self.fpga.qdrs:
            print qdr
            mqdr = myQdr.from_qdr(qdr)
            results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard,verbosity=calVerbosity)
        print 'qdr cal results:',results
        for qdrName in ['qdr0','qdr1']:
            if not results[qdr.name]:
                print 'Calibration Failed'
                break

    # calibrates QDR and initializes GbE block
    def initialize(self):
        #self.v1 = valon_synth.Synthesizer('/dev/ttyUSB4')
        #print '\n************ Valon instantiated ************'    
        #self.v1.set_frequency(0,750.0, 0.01) # LO
        #print 'Clocks set to 512 MHz'
        #self.v1.set_frequency(8,512.0, 0.01) # DAC/ADC 
        #print 'LO set to 512 MHz\n'
        
	self.dest_ip  = 192*(2**24) + 168*(2**16) + 41*(2**8) + 2 # Set to FPGA IP in /etc/network/interfaces
        self.fabric_port= 60000 
        self.fpga.write_int('tx_destip',self.dest_ip)
        self.fpga.write_int('tx_destport',self.fabric_port)
        self.fpga.write_int('sync_accum_len', self.accum_len - 1)
        self.accum_freq = self.fpga_samp_freq / self.accum_len # FPGA clock freq / accumulation length    
        self.fpga.write_int('fft_shift', 2**6 -1)    
        self.fpga.write_int('dds_shift', self.dds_shift)
        self.save_path = '/mnt/iqstream/'
        
	self.qdrCal()
	#self.initialize_GbE()
        print '\n************ QDR Calibrated ************'
        print '************ Packet streaming activated ************\n'

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
        plt.suptitle('Channel ' + str(chan),size = 20) 
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

    def freq_comb(self, freqs, samp_freq, resolution, random_phase = True, DAC_LUT = True, apply_transfunc = False):
    # Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q 
        freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res
	amp_full_scale = (2**15 - 1)
        if DAC_LUT:
	    fft_len = self.LUTbuffer_len
            bins = self.fft_bin_index(freqs, fft_len, samp_freq)
	    np.random.seed()
            phase = np.random.uniform(0., 2.*np.pi, len(bins))
            if apply_transfunc:
	    	self.amps = self.get_transfunc()	
	    else: 
	    	self.amps = np.array([1.]*len(bins))
	    if not random_phase:
	    	phase = np.load('/mnt/iqstream/last_phases.npy') 
	    self.spec = np.zeros(fft_len,dtype='complex')
	    self.spec[bins] = self.amps*np.exp(1j*(phase))
	    wave = np.fft.ifft(self.spec)
	    waveMax = np.max(np.abs(wave))
	    #wave = signal.convolve(wave,np.hanning(3), mode = 'same')
	    I = (wave.real/waveMax)*(amp_full_scale)
            Q = (wave.imag/waveMax)*(amp_full_scale)
        else:
            fft_len = (self.LUTbuffer_len/self.fft_len)
            bins = self.fft_bin_index(freqs, fft_len, samp_freq)
	    spec = np.zeros(fft_len,dtype='complex')
            amps = np.array([1.]*len(bins))
            phase = 0.
	    spec[bins] = amps*np.exp(1j*(phase))
            wave = np.fft.ifft(spec)
            #wave = signal.convolve(wave,signal.hanning(3), mode = 'same')
	    waveMax = np.max(np.abs(wave))
	    I = (wave.real/waveMax)*(amp_full_scale)
            Q = (wave.imag/waveMax)*(amp_full_scale)
	return I, Q    
    
    def select_bins(self, freqs):
    # Adjusts the DAC frequencies to the DAC frequency resolution and calculates the offset from each bin center, to be used as the DDS LUT frequencies
        bins = self.fft_bin_index(freqs, self.fft_len, self.dac_samp_freq)
        bin_freqs = bins*self.dac_samp_freq/self.fft_len
	bins[ bins < 0 ] += 1024
	
	#for i in range(len(freqs)):
	#	if (freqs[i] < 0) and (freqs[i]+ 512.0e6 >= 511.75e6):
	#		bins[i] = 1023
	
	#self.freq_residuals = np.round((freqs - bin_freqs)/self.dac_freq_res)*self.dac_freq_res
	self.freq_residuals = freqs - bin_freqs
        for i in range(len(freqs)):
		print "bin, fbin, freq, freq_res:", bins[i], bin_freqs[i]/1.0e6, freqs[i]/1.0e6, self.freq_residuals[i]
        
	ch = 0
        for fft_bin in bins:
	    self.fpga.write_int('bins', fft_bin)
            self.fpga.write_int('load_bins', 2*ch + 1)
	    self.fpga.write_int('load_bins', 0)
            ch += 1
        
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
    
    def pack_luts(self, freqs, transfunc = False):
    # packs the I and Q look-up-tables into strings of 16-b integers, in preparation to write to the QDR. Returns the string-packed look-up-tables
        if transfunc:
		self.I_dac, self.Q_dac = self.freq_comb(freqs, self.dac_samp_freq, self.dac_freq_res, random_phase = True, apply_transfunc = True)
        else:
		self.I_dac, self.Q_dac = self.freq_comb(freqs, self.dac_samp_freq, self.dac_freq_res, random_phase = True)
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
        self.I_lut_packed = self.I_lut.astype('>h').tostring()
        self.Q_lut_packed = self.Q_lut.astype('>h').tostring()
        print 'Done.'
        return 
        
    def writeQDR(self, freqs, transfunc = False):
    # Writes packed LUTs to QDR
        if transfunc:
		self.pack_luts(freqs, transfunc = True)
	else:
		self.pack_luts(freqs, transfunc = False)
        self.fpga.write_int('dac_reset',1)
        self.fpga.write_int('dac_reset',0)
        print 'Writing DAC and DDS LUTs to QDR...',
        self.fpga.write_int('start_dac',0)
        self.fpga.blindwrite('qdr0_memory',self.I_lut_packed,0)
        self.fpga.blindwrite('qdr1_memory',self.Q_lut_packed,0)
        self.fpga.write_int('start_dac',1)
        self.fpga.write_int('sync_accum_reset', 0)
        self.fpga.write_int('sync_accum_reset', 1)

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

    def read_select_bins(self):
    # Reads the snap blocks at the bin select RAM and channelizer mux
        self.fpga.write_int('chan_bins_ctrl', 0)
        self.fpga.write_int('chan_bins_ctrl', 1)
        self.chan_bins = np.fromstring(ri.fpga.read('chan_bins_bram', 8 * 2**9),dtype = '>H')
	self.chan_bins = np.hstack(zip(ri.chan_bins[2::4], self.chan_bins[3::4]))
        return

    def read_accum_snap(self):
        # Reads the avgIQ buffer. Returns I and Q as 32-b signed integers     
        self.fpga.write_int('accum_snap_ctrl', 0)
        self.fpga.write_int('accum_snap_ctrl', 1)
        accum_data = np.fromstring(self.fpga.read('accum_snap_bram', 16*2**9), dtype = '>i').astype('float')
        I0 = accum_data[0::4]    
        Q0 = accum_data[1::4]    
        I1 = accum_data[2::4]    
        Q1 = accum_data[3::4]    
        I = np.hstack(zip(I0, I1))
        Q = np.hstack(zip(Q0, Q1))
        return I, Q    

    def get_transfunc(self):
    	mag_array = np.zeros((100, len(self.test_comb)))
	for i in range(100):
		I, Q = self.read_accum_snap()
		mags = np.sqrt(I**2 + Q**2)
		mag_array[i] = mags[2:len(self.test_comb)+2]
	transfunc = np.mean(mag_array, axis = 0)
	transfunc = 1./ (transfunc / np.max(transfunc))
	np.save('./last_transfunc.npy',transfunc)
	return transfunc

    def initialize_GbE(self):
        # Configure GbE Block. Run immediately after calibrating QDR.
        self.fpga.write_int('tx_rst',0)
        self.fpga.write_int('tx_rst',1)
        self.fpga.write_int('tx_rst',0)
        return

    def stream_UDP(self, chan, Npackets):
        # CRC polynomial = [32 26 23 22 16 12 11 10 8 7 5 4 2 1 0]
	self.fpga.write_int('pps_start', 1)
        count = 0
        while count < Npackets:
            packet = self.s.recv(8234) # total number of bytes including 42 byte header
            data = np.fromstring(packet[42:],dtype = '<i').astype('float')
            data /= ( 2.0**15 - 1 )
            data /= (self.accum_len/512.)
            data *= 1000.
	    forty_two = (np.fromstring(packet[-16:-12],dtype = '>I'))
            pps_count = (np.fromstring(packet[-12:-8],dtype = '>I'))
            time_stamp = np.round((np.fromstring(packet[-8:-4],dtype = '>I').astype('float')/self.fpga_samp_freq)*1.0e3,3)
            packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
	    if (chan % 2) > 0:
                I = data[1024 + ((chan - 1) / 2)]    
                Q = data[1536 + ((chan - 1) /2)]    
            else:
                I = data[0 + (chan/2)]    
                Q = data[512 + (chan/2)]    
            phase = np.arctan2([Q],[I])
            #np.save('/mnt/iqstream/packet' + str(count) +'.npy',packet)
            print forty_two,pps_count,time_stamp,packet_count, I, Q, phase
            count += 1
	return 

    def IQ_grad(self, dark_sweep_path, plot_chan): 
	lo_freqs, I_dark, Q_dark = self.open_stored(dark_sweep_path)
	bb_freqs, delta_f = np.linspace(-200.0e6, 200.0e6, 500,retstep=True)
	#bb_freqs = np.load('/mnt/iqstream/last_bb_freqs.npy')
	channels = np.arange(len(bb_freqs))
        delta_lo = 5e3
	i_index = [np.where(np.abs(np.diff(I_dark[:,chan])) == np.max(np.abs(np.diff(I_dark[:,chan]))))[0][0] for chan in channels]
        q_index = [np.where(np.abs(np.diff(Q_dark[:,chan])) == np.max(np.abs(np.diff(Q_dark[:,chan]))))[0][0] for chan in channels]
        di_df = np.array([(I_dark[:,chan][i_index[chan] + 1] - I_dark[:,chan][i_index[chan] - 1])/(2*delta_lo) for chan in channels])
        dq_df = np.array([(Q_dark[:,chan][q_index[chan] + 1] - Q_dark[:,chan][q_index[chan] - 1])/(2*delta_lo) for chan in channels])
	I0 = np.array([I_dark[:,chan][i_index[chan]] for chan in channels])
	Q0 = np.array([Q_dark[:,chan][q_index[chan]] for chan in channels])
	rf_freqs = np.array([750.0e6 + bb_freqs[chan] for chan in channels])
	return di_df[plot_chan], dq_df[plot_chan], rf_freqs[plot_chan]
    
    def plot_stream_UDP(self, chan):
	dark_sweep_path = '/mnt/iqstream/vna_sweeps/scaled2'
       	di_df, dq_df, rf_freq = self.IQ_grad(dark_sweep_path, chan)
	Npackets = 244
	self.fpga.write_int('pps_start', 1)
        fig = plt.figure(num= None, figsize=(18,12), dpi=80, facecolor='w', edgecolor='w')
        plt.suptitle('1s stream: Channel ' + str(chan) + ', Freq = ' + str(np.round(rf_freq/1.0e6,3)) + ' MHz', fontsize = 20)
	# channel phase
	plot1 = fig.add_subplot(211)
	plot1.set_ylabel('rad')
	line1, = plot1.plot(np.arange(Npackets), np.zeros(Npackets), 'k-', linewidth = 1)
	plt.grid()
	# df
	plot2 = fig.add_subplot(212)
        plot2.set_ylabel('Hz')
	line2, = plot2.plot(np.arange(Npackets), np.zeros(Npackets), 'b-', linewidth = 1)
	plt.grid()
	plt.xlabel('Packet #', fontsize = 20)
	plt.show(block = False)
        stop = 1.0e6
	count = 0
	phases = np.zeros(Npackets)
	delta_I = np.zeros(Npackets)
	delta_Q = np.zeros(Npackets)
	df = np.zeros(Npackets)
	chan_freq = rf_freq
	while count < stop:
		packet_count = 0
		while packet_count < Npackets:
			packet = self.s.recv(8234) # total number of bytes including 42 byte header
			data = np.fromstring(packet[42:],dtype = '<i').astype('float')
			#data = np.fromstring(packet,dtype = '<i').astype('float')
			data /= 2.0**17
			data /= (self.accum_len/512.)
			ts = (np.fromstring(packet[-4:],dtype = '<i').astype('float')/ self.fpga_samp_freq)*1.0e3 # ts in ms
			if (chan % 2) > 0:
				I = data[1024 + ((chan - 1) / 2)]    
				Q = data[1536 + ((chan - 1) /2)]    
            		else:
                		I = data[0 + (chan/2)]    
                		Q = data[512 + (chan/2)]    
            		phases[packet_count] = np.arctan2([Q],[I])
			if (count and packet_count) == 0:
				I0 = I
				Q0 = Q
			delta_I = I - I0 	
			delta_Q = Q - Q0 	
       		 	df[packet_count] = ((delta_I * di_df) + (delta_Q * dq_df)) / (di_df**2 + dq_df**2)
			packet_count +=1
		avg_phase = np.round(np.mean(phases),5)
		avg_df = np.round(np.mean(df[1:]))
            	avg_dfbyf = avg_df / chan_freq
		plot1.set_ylim((np.min(phases) - 1.0e-3,np.max(phases)+1.0e-3))
            	plot2.set_ylim((np.min(df[1:]) - 1.0e-3,np.max(df[1:])+1.0e-3))
		line1.set_ydata(phases)
		line2.set_ydata(df)
        	plot1.set_title('Phase, avg =' + str(avg_phase) + ' rad', fontsize = 18)
        	plot2.set_title('Delta f, avg =' + str(avg_df) + 'Hz' + ', avg df/f = ' + str(avg_dfbyf), fontsize = 18)
		plt.draw()
		count += 1
	return 
    
    def target_sweep(self, save_path = '/mnt/iqstream/target_sweeps', write = True, span = 100.0e3):
        kid_bb_freqs = np.load('./kid_bb_freqs.npy')
        sweep_dir = raw_input('Target sweep dir ? ')
        save_path = os.path.join(save_path, sweep_dir)
        
	center_freq = 750.0e6
        
	self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
        
	kid_freqs = kid_bb_freqs +center_freq
        
	np.save('/mnt/iqstream/last_bb_freqs.npy',kid_bb_freqs)
        channels = np.arange(len(kid_freqs))
        np.save('/mnt/iqstream/last_channels.npy',channels)
        
	self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
        
	print '\nTarget baseband freqs (MHz) =', kid_bb_freqs/1.0e6
        #print '\nTarget KID freqs (MHz) =', kid_freqs/1.0e6
        
	self.writeQDR(kid_bb_freqs)
        
        self.sweep_lo(Npackets_per = 10, channels = channels, center_freq = center_freq, span = span , save_path = save_path)
        last_target_dir = save_path
        np.save('/mnt/iqstream/last_target_dir.npy',np.array([last_target_dir]))
        #self.plot_kids(save_path = last_target_dir, bb_freqs = bb_freqs, channels = channels, calc = False)
        self.plot_sweep(kid_bb_freqs, last_target_dir) 
	return

    def vna_sweep(self, center_freq = 750.0e6, save_path = '/mnt/iqstream/vna_sweeps', write = True, offset = False, find_peaks = False, find_peaks_edges = False):
        
	sweep_dir = raw_input('VNA sweep dir ? ')
        save_path = os.path.join(save_path, sweep_dir)
	bb_freqs = self.test_comb
	bb_save_path = '/mnt/iqstream/bb_freqs.npy'
		
        self.v1.set_frequency(0,center_freq / (1.0e6), 0.01) # LO
        
	print '\nVNA baseband freqs (MHz) =', bb_freqs/1.0e6
        np.save(bb_save_path,bb_freqs)
        
	self.writeQDR(bb_freqs)
        
	self.sweep_lo(Npackets_per = 100, channels = channels, center_freq = center_freq, span = delta_f, save_path = save_path)
        np.save('/mnt/iqstream/last_vna_dir.npy',np.array([save_path]))   
        self.sweep_lo(Npackets_per = 10, channels = len(bb_freqs), center_freq = center_freq, span = delta_f, save_path = save_path)
	
	get_kid_freqs(save_path)
	return 

    def sweep_lo(self, Npackets_per = 10, channels = None, center_freq = 750.0e6, span = 2.0e6, save_path = '/mnt/iqstream/lo_sweeps'):
        N = Npackets_per
        start = center_freq - (span/2.)
        stop = center_freq + (span/2.) 
        step = 2.5e3
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
            packet = self.s.recv(8234) # total number of bytes including 42 byte header
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
            packet = self.s.recv(8192 + 42) # total number of bytes including 42 byte header
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

    def get_kid_freqs(self, path):
	sweep_step = 2.5 # kHz
	smoothing_scale = 1500.0 # kHz
	peak_threshold = 0.4 # mag units
	spacing_threshold = 50.0 # kHz
	find_kids_blast.get_kids(path, ri.test_comb, sweep_step, smoothing_scale, peak_threshold, spacing_threshold)
	return
	
    def plot_sweep(self, bb_freqs, path):
        plot_sweep.plot_trace(bb_freqs, path)
	return
    
    def plot_kids(self, save_path = None, bb_freqs = None, channels = None, calc=False):
        lo_freqs, Is, Qs = self.open_stored(save_path)
        #[ plt.plot((sweep_freqs[2:] + bb_freqs[chan])/1.0e9,10*np.log10(np.sqrt(Is[:,chan][2:]**2+Qs[:,chan][2:]**2))) for chan in channels]
        mags = np.zeros((len(channels),len(lo_freqs)))
	scaled_mags = np.zeros((len(channels),len(lo_freqs)))
        for chan in range(len(channels)):
        	mags[chan] = 20*np.log10(np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2))
	if calc:
		print 'I am using the cal'
            	cal_params=np.load('/mnt/iqstream/fit_params.npy')
            	for chan in range(len(channels)):            
                	p=np.poly1d(cal_params[:,chan])                
                	mags[chan] = mags[chan]/p(lo_freqs + bb_freqs[chan])
	for chan in range(len(channels)):
		diff = 0. - np.mean(mags[chan])
		scaled_mags[chan] = diff + mags[chan]
		plt.plot((lo_freqs + bb_freqs[chan])/1.0e9,scaled_mags[chan])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('20log(mag) [dB]')
        #plt.savefig(os.path.join(save_path,'fig.png'))
        plt.show()
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
        	packet = self.s.recv(8192 + 42) # total number of bytes including 42 byte header
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

    def programLO(self, freq=750.0e6, sweep_freq=0):
        self.vi.simple_set_freq(0,freq)
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
            if opt == 0:
                os.system('clear')
                self.initialize() 
            if opt == 1:
                prompt = raw_input('Apply inverse transfer function? (y/n)')
		if prompt == 'n':
			self.writeQDR(self.test_comb, transfunc = False)
	    	if prompt == 'y':
			self.writeQDR(self.test_comb, transfunc = False)
			time.sleep(15)
			self.writeQDR(self.test_comb, transfunc = True)
	    if opt == 2:
                Npackets = input('\nNumber of UDP packets to stream? ' )
                chan = input('chan = ? ')
                self.stream_UDP(chan,Npackets)
            if opt == 3:
                self.vna_sweep(find_peaks = False)
            if opt == 4:
                self.get_kid_freqs(np.load('/mnt/iqstream/last_vna_dir.npy')[0])
            if opt == 5:
                self.target_sweep()
            if opt == 6:
                sys.exit()

        return
    
    def main(self):
        os.system('clear')
        while True: 
            self.main_opt()

if __name__=='__main__':
    ri = roachInterface()
    ri.main()
