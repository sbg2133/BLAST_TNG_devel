# This software provides a console interface to use the functions found in pipeline.py
#
# Copyright (C) 2016  Gordon, Sam <sbgordo1@asu.edu>
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
