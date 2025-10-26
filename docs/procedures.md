1. Gather hardware components: Raspberry Pi 4 Model B, NEO-8M GPS module, MPU6050 IMU module, full-size breadboard, jumper wires, status LED, 220 Ohm resistor, minimum 32GB microSD card, computer (PC or laptop), and 5V/3A power supply.
2. Gather software components: Raspberry Pi OS (latest version), Visual Studio Code, Python 3.9 or higher, and note required Python library versions (numpy, pandas, RPi.GPIO or gpiozero, etc.).
3. Create a dedicated project directory with subdirectories for code, data, and results.
4. Set up a Python virtual environment and install specified libraries with correct versions.
5. Connect the GPS module's VCC to the Raspberry Pi's 5V, the GPS module's GND to the Raspberry Pi's GND, and the GPS module's TX to the Raspberry Pi's RX for serial communication. 
6. Connect the MPU-6050's VCC pin to a 3.3V or 5V pin on the Raspberry Pi, GND to a ground pin, SDA to the Raspberry Pi's SDA (GPIO 2) pin, and SCL to the Raspberry Pi's SCL (GPIO 3) pin
7. Connect LED to Raspberry Pi via the 220 Ohm resistor. This is to show whether the device has been spoofed.
8. Boot Raspberry Pi, and enable I2C and UART interfaces using raspi-config.
9. Write and run a Python script to read and print live GPS coordinates, timestamp, and IMU accelerometer/gyroscope readings to the console for a few minutes. Verify valid data is received from both sensors.
10. Write and run a script to collect labeled training data, recording sensor readings for 30 minutes in a location with reliable GPS reception .
11. Save normal data to a structured file, automatically assigning label 0.
12. Run the data collection script again for a similar duration during a simulated spoofing attack.
13. Save simulated spoofed data to a separate structured file, assigning label 1.
14. Load normal and simulated spoofed data files with a data processing script Merge and shuffle the dataset.
15. Divide the dataset into training and testing subsets using a fixed random value. Save training and testing subsets to separate files.
16. Create a new script to define the architecture for the first neural network model, CNN only.
17. Train the first model using the training data .
18. Monitor and log performance indicators during training (observations).
19. Record total training time and transfer saved parameters to Raspberry Pi storage 
20. Develop a separate script for the second neural network architecture (CNN + EKF).
21. Train the second model using the same training dataset and stop after the same duration as the first model. Transfer the second model's parameters to the Raspberry Pi's storage .
22. Develop a testing script for the Raspberry Pi to load trained models and process live data.
23. Test the first model: read from the test subset, predict, and activate the status LED (on for spoofed, off for not spoofed) .
24. Record prediction, actual classification, and calculate confusion matrix and F1 score and accuracy score.
25. Repeat testing procedures (steps 23-24) for the second model using the same test data.
26. Repeat the entire process (steps 10-25 except 16, 20, 22; data collection, training, testing) 5 times. Gather new sets of normal and simulated spoofed data for each repetition .
27. Save all calculated evaluation metrics after each complete cycle . Consolidate all metrics into a single dataset .
28. Compute average F1 score and standard deviation for each model across all repetitions. Analyze averages and data trends.