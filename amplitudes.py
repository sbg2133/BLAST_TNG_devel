import numpy as np

def amp_to_dBm(amp):
	power = amp**2 # Watts
	dBm = 10.*np.log10(power/1.0e-3)
	print 'amp = ', amp, 'V'
	print 'power =', dBm, 'dBm'
	return 



