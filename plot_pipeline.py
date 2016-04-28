from build_menu import Console_interface
from pipeline import pipeline
ci = Console_interface()
p = pipeline()

main_prompt = '\n\t\033[35mBLAST-TNG Readout Pipeline\033[0m\n\t\033[33mChoose a number from the list and press Enter:\033[0m'                
plot_opts= ['IQ Loops','Phase scatter','Frequency noise','Phase noise']

opt = ci.mk_menu(main_prompt,plot_opts)
if opt == 0:
	chan = input('Channel = ? ')
	p.plot_loop_rotated(chan)		
if opt == 1:
	chan = input('Channel = ? ')
	p.phase_scatter(chan)
if opt == 2:
	chan = input('Channel = ? ')
	p.delta_f(chan)	
if opt == 3:
	chan = input('Channel = ? ')
	p.plot_phase_psd(chan)	
