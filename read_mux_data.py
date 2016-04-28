import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

class read_mux_data(object):
    
    def __init__(self):
        self.path = '/home/user1/blastfirmware/iqstream'
        self.option = 'target_sweeps'
        self.select = '04'
        #self.vna_path = 'vna_sweeps'
        #self.target_path = 'target_sweeps'
        self.datapath = os.path.join(self.path, self.option,self.select)
        data_files = [file for file in sorted(os.listdir(self.datapath)) if file.endswith('.npy')]
        #print files
        I = np.array([np.load(os.path.join(self.datapath,f)) for f in data_files if f.startswith('I')])
        Q = np.array([np.load(os.path.join(self.datapath,f)) for f in data_files if f.startswith('Q')])
        self.f = np.array([np.float(f[1:-4]) for f in data_files if f.startswith('I')])
        #print f
        self.z = I+1j*Q
        self.nchan = len(self.z[0])
        self.cm = plt.cm.spectral(np.linspace(0.05,0.95,self.nchan))
        self.i = self.z.real
        self.q = self.z.imag
        self.mag = np.abs(self.z)
        self.phase = np.angle(self.z)
        self.loop_centers()
        self.z_centered = self.z-self.centers
        self.rotations = np.angle(self.z_centered[self.z_centered.shape[0]/2])
        self.z_rotated = self.z_centered * np.exp(-1j*self.rotations)
        self.phase_rotated = np.angle(self.z_rotated)
        self.off_res_timestreams = np.load(os.path.join(self.datapath,'timestreams/I750.27.npy')) + 1j*np.load(os.path.join(self.datapath,'timestreams/Q750.27.npy'))
        self.on_res_timestreams = np.load(os.path.join(self.datapath,'timestreams/I750.57.npy')) + 1j*np.load(os.path.join(self.datapath,'timestreams/Q750.57.npy'))
        self.on_res_centered = self.on_res_timestreams - self.centers
        #self.off_res_centered = self.off_res_timestreams - self.centers
        self.on_res_rotated = self.on_res_centered*np.exp(-1j*self.rotations)
        #self.off_res_rotated = self.off_res_centered*np.exp(-1j*self.rotations)
        self.bb_freqs=np.load(os.path.join(self.path,'last_bb_freqs.npy'))
    
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
            #print 'fitting...',i,;sys.stdout.flush()
            (xc,yc),r = least_sq_circle_fit(chan)
            centers.append(xc+1j*yc)
        self.centers = np.array(centers)
        
    def plot_loop_shifted(self,chan):
        plt.plot(self.z_centered.real,self.z_centered.imag,color=self.cm[chan])
        plt.gca().set_aspect('equal')
        plt.xlim(np.std(self.z_centered.real)*-5,np.std(self.z_centered.real)*5)
        plt.ylim(np.std(self.z_centered.imag)*-5,np.std(self.z_centered.imag)*5)
        
    def plot_loop_rotated(self,chan):
        plt.plot(self.z_rotated.real,self.z_rotated.imag,'x',color=self.cm[chan])
        plt.gca().set_aspect('equal')
        plt.xlim(np.std(self.z_rotated.real)*-5,np.std(self.z_rotated.real)*5)
        plt.ylim(np.std(self.z_rotated.imag)*-5,np.std(self.z_rotated.imag)*5)
        
    def plot_loop_raw(self,chan):
        plt.plot(self.i[:,chan],self.q[:,chan],color=self.cm[chan])
        plt.gca().set_aspect('equal')
        
    def plot_phase_psd(self,chan):
        signal_on_mag = np.abs(self.on_res_timestreams[:,chan])
        signal_off_mag = np.abs(self.off_res_timestreams[:,chan])
        signal_on=self.on_res_rotated[:,chan]
        signal_off=self.off_res_timestreams[:,chan]
        nsamples=len(signal_on)
        time=60.
        sample_rate = nsamples/time
        time,delta_t = np.linspace(0,time,nsamples,retstep=True)
        freq,delta_f = np.linspace(0,sample_rate/2.,nsamples/2+1,retstep=True)
        rf_freq=self.bb_freqs[chan]+self.f
        
        halfway=len(self.f)/2
        plt.subplot(421)
        plt.gca().yaxis.set_major_formatter(y_formatter)
        plt.gca().xaxis.set_major_formatter(y_formatter)
        plt.plot(rf_freq/1e6,10*np.log10(self.mag[:,chan]),'r',label='swept magnitude')
        plt.plot(np.ones(len(signal_on_mag))*(self.bb_freqs[chan]+self.f[halfway])/1e6,10*np.log10(signal_on_mag),'g.',label='data on resonance')
        plt.plot(np.ones(len(signal_on_mag))*(self.bb_freqs[chan]+self.f[halfway]-300e3)/1e6,10*np.log10(signal_off_mag),'b.',label='data off resonance')
        #plt.title('phase timestream')
        plt.xlabel('Sweep Frequency [MHz]')
        plt.ylabel('10*log10(|S21|) [dB]')
        plt.legend(loc='center',fontsize='small')
        plt.grid()
        
        plt.subplot(423)
        plt.gca().yaxis.set_major_formatter(y_formatter)
        plt.gca().xaxis.set_major_formatter(y_formatter)
        plt.plot(rf_freq/1e6,self.phase_rotated[:,chan],'r',label='swept phase')
        plt.plot(np.ones(len(signal_on_mag))*(self.bb_freqs[chan]+self.f[halfway])/1e6,np.angle(signal_on),'g.',label='data on resonance')
        plt.plot(np.ones(len(signal_on_mag))*(self.bb_freqs[chan]+self.f[halfway]-300e3)/1e6,np.angle(signal_off),'b.',label='data off resonance')
        #plt.title('phase timestream')
        plt.xlabel('Sweep Frequency [MHz]')
        plt.ylabel('Phase: arctan(Q/I) [dB]')
        plt.legend(loc='center',fontsize='small')
        plt.grid()
        
        plt.subplot(222)
        plt.gca().set_aspect('equal','datalim')
        plt.gca().yaxis.set_major_formatter(y_formatter)
        plt.plot(self.z_rotated[:,chan].real,self.z_rotated[:,chan].imag,'r',marker='x',label='swept iq loop')
        plt.plot(signal_on.real, signal_on.imag,'g.',label='data on resonance')
        plt.plot(signal_off.real,signal_off.imag,'b.',label='data off resonance')
        #plt.title('IQ loop')
        plt.xlabel('I [arb. units]')
        plt.ylabel('Q [arb. units]')
        plt.legend(loc='center',fontsize='small')
        plt.grid()
        
        plt.subplot(212)
        psd_on  = delta_t/nsamples*np.abs(np.fft.rfft(np.angle(signal_on)))**2
        psd_off = delta_t/nsamples*np.abs(np.fft.rfft(np.angle(signal_off)))**2
        plt.loglog(freq,psd_on,'g',label='noise on resonance')
        plt.loglog(freq,psd_off,'b',label='noise off resonance')
        plt.xlim(0.01,200)
        plt.ylim(1e-13,1e-3)
        #plt.title('phase noise')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [rad^2 / Hz]')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.suptitle('Blast-TNG 250um array   2016-01-31   Channel %03d/%d'%(chan,self.nchan),fontsize='large')
        
d=read_mux_data()
"""
plt.figure(figsize=(14,12))
for i in range(d.nchan):
    d.plot_phase_psd(i)
    plt.savefig(os.path.join(d.datapath,'figures','psd%04drotated.png'%i))
    #plt.savefig(os.path.join(d.datapath,'figures','psd%04d.svg'%i))
    plt.clf()
    print ' plotting ',i,;sys.stdout.flush()
#d.plot_psd_on_res(0)
plt.show()
"""
