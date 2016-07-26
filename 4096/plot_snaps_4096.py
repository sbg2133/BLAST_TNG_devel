# This software provides a console interface to pull data from the BLAST-TNG 
# ROACH2 firmware. It should be run after one of the 'loopback' scripts, in a seperate console. 

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

from read_snaps_4096 import FirmwareSnaps
import matplotlib.pyplot as plt
fs = FirmwareSnaps()

main_prompt = '\n\t\033[35mROACHII mKID Readout\033[0m\n\t\033[33mChoose a number from the list and press Enter:\033[0m'                
plot_opts= ['Plot ADC stream','Plot FFT stream','Plot DDS Mixer Channel','Plot Accumulated Channel Magnitudes', 'Stream channel phase', 'Read accum reg']

opt = fs.menu(main_prompt,plot_opts)
if opt == 0:
	fs.plotADC()
if opt == 1:
	fs.plotFFT()
if opt == 2:
	chan = input('Channel = ? ')
	fs.plotMixer(chan)
if opt == 3:
	fs.plotAccum()
if opt == 4:
	chan = input('Channel = ? ')
	fs.plotPhase(chan)
if opt == 5:
	fs.read_accum_reg()	

