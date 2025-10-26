from mpu6050.mpu6050 import mpu6050
import time

sensor = mpu6050(0x68)

try:
    while True:
        accel = sensor.get_accel_data()
        gyro = sensor.get_gyro_data()
        temp = sensor.get_temp()
        print(f"Accel: X:{accel['x']:.2f}, Y:{accel['y']:.2f}, Z:{accel['z']:.2f}")
        print(f"Gyro: X:{gyro['x']:.2f}, Y:{gyro['y']:.2f}, Z:{gyro['z']:.2f}")
        print(f"Temp: {temp:.2f} �C")
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped reading sensor data.")
