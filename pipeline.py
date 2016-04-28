import os,sys
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import optimize
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

class pipeline(object):
    
	def __init__(self):
		self.path = '/home/muchacho/iqstream'
		self.option = 'target_sweeps'
		self.select = '04'
		#self.vna_path = 'vna_sweeps'
		#self.target_path = 'target_sweeps'
		self.datapath = os.path.join(self.path, self.option,self.select)
		data_files=[f for f in sorted(os.listdir(self.datapath)) if f.endswith('.npy')]
		I = np.array([np.load(os.path.join(self.datapath,f)) for f in data_files if f.startswith('I')])
		Q = np.array([np.load(os.path.join(self.datapath,f)) for f in data_files if f.startswith('Q')])
		self.lo_freqs = np.array([np.float(f[1:-4]) for f in data_files if f.startswith('I')])
		#print f
		self.chan_ts = I + 1j*Q
		self.nchan = len(self.chan_ts[0])
		self.cm = plt.cm.spectral(np.linspace(0.05,0.95,self.nchan))
		self.i = self.chan_ts.real
		self.q = self.chan_ts.imag
		self.mag = np.abs(self.chan_ts)
		self.phase = np.angle(self.chan_ts)
		self.loop_centers() # returns self.centers
		self.chan_ts_centered = self.chan_ts - self.centers
		self.rotations = np.angle(self.chan_ts_centered[self.chan_ts_centered.shape[0]/2])
		self.chan_ts_rotated = self.chan_ts_centered * np.exp(-1j*self.rotations)
		self.phase_rotated = np.angle(self.chan_ts_rotated)
		self.ts_off = np.load(os.path.join(self.datapath,'timestreams/I750.27.npy')) + 1j*np.load(os.path.join(self.datapath,'timestreams/Q750.27.npy'))
		self.ts_on = np.load(os.path.join(self.datapath,'timestreams/I750.57.npy')) + 1j*np.load(os.path.join(self.datapath,'timestreams/Q750.57.npy'))
        	self.ts_on_centered = self.ts_on - self.centers
        	self.ts_on_rotated = self.ts_on_centered *np.exp(-1j*self.rotations)
        	self.bb_freqs=np.load(os.path.join(self.path,'last_bb_freqs.npy'))
		self.i_off, self.q_off = self.ts_off.real, self.ts_off.imag
		self.i_on, self.q_on = self.ts_on_rotated.imag, self.ts_on_rotated.imag
		self.phase_off = np.angle(self.ts_off)	
		self.phase_on = np.angle(self.ts_on_rotated)	
		self.kid_freqs=np.load(os.path.join(self.path,'last_kid_freqs.npy'))
		self.bb_freqs=np.load(os.path.join(self.path,'last_bb_freqs.npy'))
		self.rf_freqs=np.load(os.path.join(self.path,'last_rf_freqs.npy'))
		self.delta_lo = 2.5e3
		
	def phase_scatter(self,chan):
		fig = plt.figure()
		plt.suptitle('Phase scatter, Channel = ' + str(chan))
		plot1 = plt.subplot(1,2,1)
		plot1.scatter(self.i_off[:,chan], self.q_off[:,chan], color = 'b', label = 'off res')	
		plot1.scatter(self.i_on[:,chan], self.q_on[:,chan], color = 'r', label = 'on res')
		plt.xlabel('I')
		plt.ylabel('Q')
		plot1.set_autoscale_on(True)
		off_data = sorted(self.phase_off[:,chan])
		on_data = sorted(self.phase_on[:,chan])
		fwhm_off = np.abs(np.round(2.355*np.std(self.phase_off[:,chan]),3))
		fwhm_on = np.abs(np.round(2.355*np.std(self.phase_on[:,chan]),3))
		off_fit = stats.norm.pdf(off_data, np.mean(self.phase_off[:,chan]), np.std(self.phase_off[:,chan]))
		on_fit = stats.norm.pdf(on_data, np.mean(self.phase_on[:,chan]), np.std(self.phase_on[:,chan]))
		plot2 = plt.subplot(1,2,2)
		plot2.plot(off_data - np.mean(off_data), off_fit, color = 'b', label = 'fwhm off = '+ str(fwhm_off))
		plot2.plot(on_data - np.mean(on_data), on_fit, color = 'r', label = 'fwhm on = '+ str(fwhm_on))
		plt.xlabel('rad')
		plt.ylabel('Prob. Density')
		plt.legend()
		plt.show()
		return
	
	def delta_f(self, channel):	
		i_index = [np.where(np.abs(np.diff(self.i[:,chan])) == np.max(np.abs(np.diff(self.i[:,chan]))))[0][0] for chan in range(self.nchan)]
		q_index = [np.where(np.abs(np.diff(self.q[:,chan])) == np.max(np.abs(np.diff(self.q[:,chan]))))[0][0] for chan in range(self.nchan)]
		self.di_df = np.array([(self.i[:,chan][i_index[chan] + 1] - self.i[:,chan][i_index[chan] - 1])/(2*self.delta_lo) for chan in range(self.nchan)])
		self.dq_df = np.array([(self.q[:,chan][q_index[chan] + 1] - self.q[:,chan][q_index[chan] - 1])/(2*self.delta_lo) for chan in range(self.nchan)])
		self.delta_i_on = [self.i_on[:,chan] - np.mean(self.i_on[:,chan]) for chan in range(self.nchan)] 
		self.delta_q_on = [self.q_on[:,chan] - np.mean(self.q_on[:,chan]) for chan in range(self.nchan)] 
		self.delta_i_off = [self.i_off[:,chan] - np.mean(self.i_off[:,chan]) for chan in range(self.nchan)] 
		self.delta_q_off = [self.q_off[:,chan] - np.mean(self.q_off[:,chan]) for chan in range(self.nchan)] 
		self.df_on = [ ((self.delta_i_on[chan] * self.di_df[chan]) + (self.delta_q_on[chan] * self.dq_df[chan]) / (self.di_df[chan]**2 + self.dq_df[chan]**2)) for chan in range(self.nchan)]
		self.df_off = [ ((self.delta_i_off[chan] * self.di_df[chan]) + (self.delta_q_off[chan] * self.dq_df[chan]) / (self.di_df[chan]**2 + self.dq_df[chan]**2)) for chan in range(self.nchan)]
		time = np.arange(0, len(self.i_off))/244.
		frequ = np.arange(1, len(self.i_off)+1)*244./len(self.i_off)
		plt.plot(np.log10(frequ), np.log10((np.abs(np.fft.fft(self.df_off[channel]/self.kid_freqs[channel])))**2/len(self.i_off)), label = r'$\Delta$f off', color = 'black')
		plt.plot(np.log10(frequ), np.log10((np.abs(np.fft.fft(self.df_on[channel]/self.kid_freqs[channel])))**2/len(self.i_off)), label = r'$\Delta$f on', color = 'red', alpha = 0.5)
		#plt.plot(self.df_off[channel], color = 'b', label = 'off')
		#plt.plot(self.df_on[channel], color= 'g', label = 'on')
		plt.title(r'$\Delta$f, Channel = ' + str(channel))
		plt.xlabel('log$_{10}$ freq (Hz)')
		plt.ylabel(r'log$_{10}$ (ef/f$_{0}$)$^{2}$ (Hz)')
		plt.legend()
		plt.show()
		return 

	def loop_centers(self):
        	def least_sq_circle_fit(chan):
			"""
			Least squares fitting of circles to a 2d data set. 
			Calcultes jacobian matrix to speed up scipy.optimize.least_sq. 
			Complements to scipy.org
			Returns the center and radius of the circle ((xc,yc), r)
			"""
			x=self.i[:,chan]
			y=self.q[:,chan]
			xc_guess = x.mean()
			yc_guess = y.mean()
                        
			def calc_radius(xc, yc):
				""" calculate the distance of each data points from the center (xc, yc) """
				return np.sqrt((x-xc)**2 + (y-yc)**2)

			def f(c):
				""" calculate f, the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
				Ri = calc_radius(*c)
				return Ri - Ri.mean()

			def Df(c):
				""" Jacobian of f.The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
				xc, yc = c
				dfdc = np.empty((len(c), x.size))
				Ri = calc_radius(xc, yc)
				dfdc[0] = (xc - x)/Ri            # dR/dxc
				dfdc[1] = (yc - y)/Ri            # dR/dyc
				dfdc = dfdc - dfdc.mean(axis=1)[:, np.newaxis]
				return dfdc
			
			(xc,yc), success = optimize.leastsq(f, (xc_guess, yc_guess), Dfun=Df, col_deriv=True)
		    
			Ri = calc_radius(xc,yc)
			R = Ri.mean()
			residual = sum((Ri - R)**2)
			#print xc_guess,yc_guess,xc,yc
			return (xc,yc),R

        	centers=[]
        	for chan in range(self.nchan):
            		(xc,yc),r = least_sq_circle_fit(chan)
            		centers.append(xc+1j*yc)
        	self.centers = np.array(centers)
		return

	def plot_loop_centered(self,chan):
		plt.plot(self.chan_ts_centered.real[:,chan],self.chan_ts_centered.imag[:,chan],'x',color=self.cm[chan])
		plt.gca().set_aspect('equal')
		plt.xlim(np.std(self.chan_ts_centered.real[:,chan])*-3,np.std(self.chan_ts_centered.real[:,chan]*3))
		plt.ylim(np.std(self.chan_ts_centered.imag[:,chan])*-3.,np.std(self.chan_ts_centered.imag[:,chan]*3))
		plt.show()
		return

	def plot_loop_rotated(self,chan):
		plt.title('IQ loop Channel = ' + str(chan) + ', centered and rotated')
		plt.plot(self.chan_ts_rotated.real[:,chan],self.chan_ts_rotated.imag[:,chan],'x',color='black')
		plt.gca().set_aspect('equal')
		plt.xlim(np.std(self.chan_ts_rotated.real[:,chan])*-3,np.std(self.chan_ts_rotated.real[:,chan])*3)
		plt.ylim(np.std(self.chan_ts_rotated.imag[:,chan])*-3,np.std(self.chan_ts_rotated.imag[:,chan])*3)
		plt.xlabel('I')
		plt.ylabel('Q')
		plt.show()
		return

	def plot_phase_psd(self,chan):
		sig_on_mag = np.abs(self.ts_on[:,chan])
		sig_off_mag = np.abs(self.ts_off[:,chan])
		sig_on = self.ts_on_rotated[:,chan]
		sig_off = self.ts_off[:,chan]
		nsamples = len(sig_on)
		time = 60.
		sample_rate = nsamples/time
		time,delta_t = np.linspace(0,time,nsamples,retstep=True)
		freq,delta_f = np.linspace(0,sample_rate/2.,nsamples/2+1,retstep=True)
		rf_freq = self.bb_freqs[chan]+self.lo_freqs

		halfway = len(self.lo_freqs)/2
		plt.figure(figsize=(14,12))
		plt.subplot(421)
        	plt.gca().yaxis.set_major_formatter(y_formatter)
        	plt.gca().xaxis.set_major_formatter(y_formatter)
		plt.plot(rf_freq/1e6,10*np.log10(self.mag[:,chan]),'r',label='swept magnitude')
		plt.plot(np.ones(len(sig_on_mag))*(self.bb_freqs[chan]+self.lo_freqs[halfway])/1e6,10*np.log10(sig_on_mag),'black',label='on res')
		plt.plot(np.ones(len(sig_on_mag))*(self.bb_freqs[chan]+self.lo_freqs[halfway]-300e3)/1e6,10*np.log10(sig_off_mag),'b.',label='off res')
		plt.xlabel('Sweep Frequency [MHz]')
		plt.ylabel('10log$_{10}$(|S21|) [dB]')
		plt.legend(loc = 'center',fontsize = 'small')
		plt.grid()

		plt.subplot(423)
        	plt.gca().yaxis.set_major_formatter(y_formatter)
        	plt.gca().xaxis.set_major_formatter(y_formatter)
		plt.plot(rf_freq/1e6,self.phase_rotated[:,chan],'r',label='swept phase')
		plt.plot(np.ones(len(sig_on_mag))*(self.bb_freqs[chan]+self.lo_freqs[halfway])/1e6,np.angle(sig_on),'black',label='on res')
		plt.plot(np.ones(len(sig_on_mag))*(self.bb_freqs[chan]+self.lo_freqs[halfway]-300e3)/1e6,np.angle(sig_off),'b.',label=' off res')
		plt.xlabel('Sweep Frequency [MHz]')
		plt.ylabel('Phase: arctan(Q/I) [dB]')
		plt.legend(loc='center',fontsize='small')
		plt.grid()
		plt.subplot(222)
		plt.gca().set_aspect('equal','datalim')
        	plt.gca().yaxis.set_major_formatter(y_formatter)
		plt.plot(self.chan_ts_rotated[:,chan].real,self.chan_ts_rotated[:,chan].imag,'r',marker='x',label='swept IQ loop')
		plt.plot(sig_on.real, sig_on.imag,'black',label='on res')
		plt.plot(sig_off.real,sig_off.imag,'b',label='off res')
		plt.xlabel('I [arb. units]')
		plt.ylabel('Q [arb. units]')
		plt.legend(loc='lower left',fontsize='small')
		plt.grid()

		plt.subplot(212)
		psd_on  = delta_t/nsamples*np.abs(np.fft.rfft(np.angle(sig_on)))**2
		psd_off = delta_t/nsamples*np.abs(np.fft.rfft(np.angle(sig_off)))**2
		plt.loglog(freq,psd_on,'r',label='noise on')
		plt.loglog(freq,psd_off,'black',alpha = 0.5, label='noise off')
		plt.xlim(0.01,200)
		plt.ylim(1e-13,1e-3)
		plt.xlabel('Freq [Hz]')
		plt.ylabel(r'PSD [rad$^{2}$ / Hz]')
		plt.legend()
		plt.grid()
		plt.tight_layout()
		plt.subplots_adjust(top=0.95)
		plt.suptitle('Blast-TNG 250um array   2016-01-31   Channel %03d/%d'%(chan,self.nchan),fontsize='large')
		plt.show()
		return
