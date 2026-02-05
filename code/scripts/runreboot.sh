#!/bin/bash
cd /home/pi/GPS-Spoofing-Detection || exit 1

. /home/pi/GPS-Spoofing-Detection/myvenv/bin/activate

python /home/pi/GPS-Spoofing-Detection/code/test/led.py
