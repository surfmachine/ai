# Introduction

## Setting up the Raspberry Pi

Download Raspberry Imager to install latest Raspberry OS on SD card

Follow https://zedt.eu/tech/linux/how-to-pre-configure-raspberry-pi-for-remote-ssh-and-wifi/

Create an empty file called shh

Create wpa_supplicant.conf

```
country=CH
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="NETWORK-NAME"
    psk="NETWORK-PASSWORD"
}
```

Find the IP Adress of the raspberry

ssh pi@192.168.0.24

password is raspberry

Configure the Raspberry

```
sudo raspi-config
```
* Change User Password
* Set timezone (Localisation Options)
* Enable I2C (Interfacing Options)
* Expand the file system (Advanced Options)
* Reboot once the changes are made


Locale Settings
If a Perl error message appears warning about locale settings, check available locales
```
locale -a
```
Then either install a new locale or check spelling of existing ones and set
```
export LANGUAGE=en_GB.utf8
export LC_ALL=en_GB.utf8
export LC_CTYPE=en_GB.utf8
export LANG=en_GB.utf8
```

Step 11: Get the updates
```
sudo apt-get update -y
sudo apt-get upgrade -y
```

Step 12: Install vim
```
sudo apt-get install vim
```
How to use vim:
* `vim filename.extension`
* [I] (to insert text)
* [esc] (to stop inserting)
* :q to quit or :wq to write & quit

Step 13: Change hostname
```
sudo vim /etc/hostname
sudo vim /etc/hosts
```
Change raspberrypi to something else

Step 14: Install I2C tools
```
sudo apt-get install -y python-smbus
sudo apt-get install -y i2c-tools
```
Check that the tools are installed
```
sudo i2cdetect -y 1
```

Step 15: Python Settings
Raspbian comes with two versions of Python:
```
python --version
python3 --version
```
To set a specific version as default
```
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2
```
Now calling `python` will automatically use Python 3.7
