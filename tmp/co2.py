import serial

#inspired from c code of http://www.seeedstudio.com/wiki/Grove_-_CO2_Sensor

inp = []
cmd_zero_sensor = b'\xff\x87\x87\x00\x00\x00\x00\x00\xf2'
cmd_span_sensor = b'\xff\x87\x87\x00\x00\x00\x00\x00\xf2'
cmd_get_sensor = b'\xff\x01\x86\x00\x00\x00\x00\x00\x79'
    
ser = serial.Serial('/dev/serial0', 9600)	#Open the serial port at 9600 baud
ser.flush()
        
ser.write(cmd_get_sensor)
inp = ser.read(9)
high_level = inp[2]
low_level = inp[3]
temp_co2  = inp[4] - 40

#output in ppm
conc = high_level*256+low_level
print(conc)