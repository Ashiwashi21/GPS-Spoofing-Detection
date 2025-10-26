# testing the led by setting up a circuit and making it flash 50 times with sleep times of 0.5 seconds
from gpiozero import LED
import time

led = LED(17)

print("Starting LED test...")

for i in range(20):
    led.on()
    print(f"Cycle {i+1}: LED ON")
    time.sleep(0.3)
    led.off()
    print(f"Cycle {i+1}: LED OFF")
    time.sleep(0.3)

print("LED test complete.")
