import os, sys, time, select, termios, tty, ftdi, subprocess, serial, logging, telnetlib, shutil
from optparse import OptionParser
from mpsse import *
import i2c_functions as iicf
import xmodem_tx as xtx
import defs_max16071, defs_max1805, defs_ad7414
import defs_r2_ats as defs
REV = 2

# The I2C bus has 2 masters, FTDI and PPC. This method will check the state of the PPC and wait till it has released the I2C bus.
# This method opens a serial port object, if the serial port is already open that object wont be affected.
# Note that the delays are multiplied by 10 as the serial port timeout is set to 0.1s
def check_ppc_i2c():
  return True
  try:
    serial_obj = open_ftdi_uart(ser_port, baud, 0.1)
  except:
    raise
  try:
    serial_obj.flushInput()
    serial_obj.flushOutput()
    i2c_avbl = False
    if find_str_ser(serial_obj, 'DRAM:', 8, False)[0]:
      if find_str_ser(serial_obj, 'stop autoboot:', defs.UBOOT_DELAY*10, False)[0]:
        serial_obj.write('\n')
        i2c_avbl = True
      else:
        raise Exception('ERROR: U-Boot did not load correctly during checking that the PPC released the I2C bus.')
    # Check if there is activity on the serial port, if so Linux is booting.
    elif (len(serial_obj.read(20)) > 0):
      print '    Waiting for Linux to boot, this will take about a minute.' 
      tout = 0
      while (not i2c_avbl) and (tout < defs.BOOT_DELAY*10):
        serial_obj.write('\n')
        if find_str_ser(serial_obj, '=>', 1*10, False)[0]:
          i2c_avbl = True
        else:
          serial_obj.write('\n')
          if find_str_ser(serial_obj, 'login:', 1*10, False)[0]:
            i2c_avbl = True
        tout += 1
      if not i2c_avbl:
        raise Exception('ERROR: Timed out waiting for Linux to boot.')
    else:
      i2c_avbl = True
    return i2c_avbl
  finally:
    serial_obj.flushInput()
    serial_obj.flushOutput()
    serial_obj.close()

def open_ftdi_b():
  try:
    i2c_bus = MPSSE()
    i2c_bus.Open(defs.R2_VID, defs.R2_PID, I2C, ONE_HUNDRED_KHZ, MSB, IFACE_B, None)
    return i2c_bus
  except Exception as e:
    if e.message == 'device not found':
      print '\nERROR: Could not connect to I2C bus, check USB connection.'
    raise

def read_voltage(vbus, i2c_bus):
  ch = defs.V_MON_MAP[vbus][0]
  addr = defs.V_MON_MAP[vbus][1]
  ch_idx = (ch - 1)*2
  msb = defs_max16071.CH_ARRAY[ch_idx]
  lsb = defs_max16071.CH_ARRAY[ch_idx+1]
  voltage = ((iicf.i2c_regread(i2c_bus, addr , msb)) << 2) + ((iicf.i2c_regread(i2c_bus, addr, lsb)) >> 6)
  # get full scale voltage for requested channel
  if ch < 5:
    adc_conf = iicf.i2c_regread(i2c_bus, addr, defs_max16071.ADC_CONF_4321)
    fs_v = defs_max16071.ADC_RNG[(adc_conf >> ch_idx) & 0x03]
  else:
    adc_conf = iicf.i2c_regread(i2c_bus, addr, defs_max16071.ADC_CONF_8765)
    fs_v = defs_max16071.ADC_RNG[(adc_conf >> (ch_idx - 8)) & 0x03]
  # if 12V monitor apply voltage divider
  if vbus == '12V':
    volts = (voltage/1024.0)*fs_v*defs_max16071.V_DIV_12V
  else:
    volts = (voltage/1024.0)*fs_v
  return volts

def read_current(vbus, i2c_bus):
  ch = defs.C_MON_MAP[vbus]
  ch_idx = (ch - 1)*2
  msb = defs_max16071.CH_ARRAY[ch_idx]
  lsb = defs_max16071.CH_ARRAY[ch_idx+1]
  addr = iicf.ADDR_C_MON
  voltage = ((iicf.i2c_regread(i2c_bus, addr , msb)) << 2) + ((iicf.i2c_regread(i2c_bus, addr, lsb)) >> 6)
  # get full scale voltage for requested channel
  if ch < 5:
    adc_conf = iicf.i2c_regread(i2c_bus, addr, defs_max16071.ADC_CONF_4321)
    fs_v = defs_max16071.ADC_RNG[(adc_conf >> ch_idx) & 0x03]
  else:
    adc_conf = iicf.i2c_regread(i2c_bus, addr, defs_max16071.ADC_CONF_8765)
    fs_v = defs_max16071.ADC_RNG[(adc_conf >> (ch_idx - 8)) & 0x03]
  res = defs_max16071.SNS_RES[ch - 1]
  # Select gain resistor according to ROACH2 revision
  if REV == 1:
    gain = 1 + (1e5/defs_max16071.GAIN_RES_REV1[ch - 1])
  else:
    gain = 1 + (1e5/defs_max16071.GAIN_RES_REV2[ch - 1])
  amps = ((voltage/1024.0)*fs_v)/(res*gain)
  return amps

# read max16071 onboard current
def read_ob_current(vbus, i2c_bus):
  if vbus == '5V0':
    addr = iicf.ADDR_C_MON
    res = defs_max16071.SNS_RES_5V0
  else: #12V current
    addr = iicf.ADDR_V_MON
    res = defs_max16071.SNS_RES_12V
  voltage = iicf.i2c_regread(i2c_bus, addr, defs_max16071.MON_C)
  #Get current-sense gain setting
  curr_sns_gn = defs_max16071.CURR_SNS_GAIN[((iicf.i2c_regread(i2c_bus, addr, defs_max16071.OCPT_CSC1)) >> 2) & 0x03]
  amps = (((voltage/255.0)*1.4)/curr_sns_gn)/res
  return amps

def read_vmon_gpio(gpio):
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      ch = defs.V_MON_GPIO[gpio] - 1
      addr = iicf.ADDR_V_MON
      gpio_rd = iicf.i2c_regread(i2c_bus, addr, defs_max16071.GPIO_INPUT_STATE)
      val = (gpio_rd >> ch) & 0x01
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), voltage monitor gpio not read.')
  return val

def check_currents(dic):
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      c_err = []
      for i, v in dic.iteritems():
        vbus = i[:-2]
        if (vbus == '12V') | (vbus == '5V0'):
          amps = read_ob_current(vbus, i2c_bus)
        else:
          amps = read_current(vbus, i2c_bus)
        if i[-1] == 'H':
          if amps > v:
            c_err.append(i)
            c_err.append(amps)
        else:
          if amps < v:
            c_err.append(i)
            c_err.append(amps)
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), currents not read.')
  return c_err

def check_voltages():
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      v_err = []
      # Select voltage profile according to ROACH2 Revision
      if REV == 1:
        dic = defs.V_THRESHOLD_REV1
      else:
        dic = defs.V_THRESHOLD_REV2
      for i, v in dic.iteritems():
        volts = read_voltage(i[:-2], i2c_bus)
        if i[-1] == 'H':
          if volts > v:
            v_err.append(i)
            v_err.append(volts)
        else:
          if volts < v:
            v_err.append(i)
            v_err.append(volts)
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), voltages not read.')
  return v_err

def print_v_c():
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      print "    1V0 Bus: %.3fV, %.3fA" %(read_voltage('1V0', i2c_bus), read_current('1V0', i2c_bus))
      print "    1V5 Bus: %.3fV, %.3fA" %(read_voltage('1V5', i2c_bus), read_current('1V5', i2c_bus))
      print "    1V8 Bus: %.3fV, %.3fA" %(read_voltage('1V8', i2c_bus), read_current('1V8', i2c_bus))
      print "    2V5 Bus: %.3fV, %.3fA" %(read_voltage('2V5', i2c_bus), read_current('2V5', i2c_bus))
      print "    3V3 Bus: %.3fV, %.3fA" %(read_voltage('3V3', i2c_bus), read_current('3V3', i2c_bus))
      print "    5V0 Bus: %.3fV, %.3fA" %(read_voltage('5V0', i2c_bus), read_ob_current('5V0', i2c_bus))
      if REV == 1:
        # MAX16071 mod puts 12v line on 3v3aux monitor
        print "    12V Bus: %.3fV, %.3fA" %((read_voltage('3V3_AUX', i2c_bus)*defs_max16071.V_DIV_12V), read_ob_current('12V', i2c_bus))
        print "    3V3 Aux: not avialable on revision 1"
      else:
        print "    12V Bus: %.3fV, %.3fA" %(read_voltage('12V', i2c_bus), read_ob_current('12V', i2c_bus))
        print "    3V3 Aux: %.3fV" %read_voltage('3V3_AUX', i2c_bus)
        print "    5V0 Aux: %.3fV" %read_voltage('5V0_AUX', i2c_bus)
      print "    MGT 1.2V Power Good = %d" %read_vmon_gpio('MGT_1V2_PG')
      print "    MGT 1.0V Power Good = %d" %read_vmon_gpio('MGT_1V0_PG')
      print ""
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), voltages and currents not read.')

def read_temp(sensor, i2c_bus):
  if sensor == 'PPC':
    temp = iicf.i2c_regread(i2c_bus, iicf.ADDR_PPC_FPGA_TEMP, defs_max1805.RET1)
  elif sensor == 'FPGA':
    temp = iicf.i2c_regread(i2c_bus, iicf.ADDR_PPC_FPGA_TEMP, defs_max1805.RET2)
  elif sensor == 'INLET':
    temp_reg = iicf.i2c_regread2b(i2c_bus, iicf.ADDR_INLET_TEMP, defs_ad7414.TEMP)
    temp = ((temp_reg[0]<<2)+(temp_reg[1]>>6))*0.25
  elif sensor == 'OUTLET':
    temp_reg = iicf.i2c_regread2b(i2c_bus, iicf.ADDR_OUTLET_TEMP, defs_ad7414.TEMP)
    temp = ((temp_reg[0]<<2)+(temp_reg[1]>>6))*0.25
  return temp

def check_temps():
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      t_err = []
      for i, v in defs.T_THRESHOLD.iteritems():
        temp = read_temp(i[:-4], i2c_bus)
        if i[-1] == 'H':
          if temp > v:
            t_err.append(i)
            t_err.append(temp)
        else:
          if temp < v:
            t_err.append(i)
            t_err.append(temp)
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), temperatures not read.')
  return t_err

def print_temps():
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      print '    PPC Temp: %d degreesC' %read_temp('PPC', i2c_bus)
      print '    FPGA Temp: %d degreesC' %read_temp('FPGA', i2c_bus)
      print '    Inlet Temp: %0.2f degreesC' %read_temp('INLET', i2c_bus)
      print '    Inlet Temp: %0.2f degreesC' %read_temp('OUTLET', i2c_bus)
      print ''
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), temperatures not read.')

def read_vmon_gpio(gpio):
  if check_ppc_i2c():
    try:
      i2c_bus = open_ftdi_b()
    except:
      raise
    try:
      ch = defs.V_MON_GPIO[gpio] - 1
      addr = iicf.ADDR_V_MON
      gpio_rd = iicf.i2c_regread(i2c_bus, addr, defs_max16071.GPIO_INPUT_STATE)
      val = (gpio_rd >> ch) & 0x01
    finally:
      i2c_bus.Close()
  else:
    raise Exception('ERROR: I2c bus could not be secured from the PPC (check PPC state), voltage monitor gpio not read.')
  return val


