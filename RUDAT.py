# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:26:41 2015
Python Module for controlling the RUDAT variable attenuators via COM port mode
@author: Brad Dober
"""
import struct
import serial

class Attenuator:
    def __init__(self,port):
        self.conn = serial.Serial(None, 9600, serial.EIGHTBITS,
                                  serial.PARITY_NONE, serial.STOPBITS_ONE)
        self.conn.setPort(port)
        
    def get_serial(self):
        self.conn.open()
        self.conn.write('\rS\r')
        serial=self.conn.readline()
        serial=int(serial)
        self.conn.close()
        return serial
        
    def get_model(self):
        self.conn.open()
        self.conn.write('\rM\r')
        model=self.conn.readline()
        model=model.strip()
        self.conn.close()
        return model       
        
    def get_atten(self):
        """
        Returns the current attenuation setting in dBm
        """
        self.conn.open()
        self.conn.write('\rA\r')
        atten = self.conn.readline()
        atten =  float(atten)
        self.conn.close()
        return atten
        
    def set_atten(self,atten):
        """
        Set the desired attenuation in dBm
        """
        b_atten='\rB' + str(atten) +'E\r'
        self.conn.open()
        self.conn.write(b_atten)
        result=self.conn.readline()
        if result != 'ACK\r\n':
            print 'Command not registered! Try again'
        else:
            print 'ok'
        self.conn.close()
    