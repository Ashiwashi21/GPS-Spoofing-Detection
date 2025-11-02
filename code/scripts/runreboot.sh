#!/bin/bash
cd /home/pi/GPS-Spoofing-Detection || exit 1
source /home/pi/GPS-Spoofing-Detection/myvenv/bin/activate
/home/pi/GPS-Spoofing-Detection/myvenv/bin/python code/scripts/datacollection.py
