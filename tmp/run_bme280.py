import smbus2
import bme280 # https://pypi.org/project/RPi.bme280/
from datetime import datetime

port = 1
address = 0x76
bus = smbus2.SMBus(port)

bme280.load_calibration_params(bus, address)
data = bme280.sample(bus, address)

try:
    f = open("/home/pi/home-automation/bme280.txt", "x")
    f.write("date" + "\t" + "time" + "\t" + "temperature" + "\t" + "humidity" + "\t" + "pressure" + "\n")
    f.close()
except:
    pass
	
temperature = str(round(data.temperature, 2))
humidity = str(round(data.humidity, 2))
pressure = str(round(data.pressure, 2))

now = datetime.now()
date = now.strftime("%Y-%m-%d")
time = now.strftime("%H:%M:%S")

print(date + "\t" + time + "\t" + temperature + "\t" + humidity + "\t" + pressure + "\n")

#f = open("/home/pi/home-automation/bme280.txt", "a+")
#f.write(date + "\t" + time + "\t" + temperature + "\t" + humidity + "\t" + pressure + "\n")
#f.close()
