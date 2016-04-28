from loopback_0425 import roachInterface 
ri = roachInterface()

main_prompt = '\n\t\033[35mROACHII mKID Readout\033[0m\n\t\033[33mChoose a number from the list and press Enter:\033[0m'                
plot_opts= ['Plot ADC stream','Plot FFT stream','Plot DDS Mixer Channel','Plot Accumulated Channel Magnitudes', 'Stream channel phase', 'Plot PSD']

opt = ri.menu(main_prompt,plot_opts)
if opt == 0:
	ri.plotADC()
if opt == 1:
	ri.plotFFT()
if opt == 2:
	chan = input('Channel = ? ')
	ri.plotMixer(chan)
if opt == 3:
	ri.plotAccum()
if opt == 4:
	chan = input('Channel = ? ')
	ri.plotPhase(chan)
if opt == 5:
	chan = input('Channel = ? ')
	time_interval = input('Time interval (s) = ? ')
	ri.plotPSD(chan, time_interval)

