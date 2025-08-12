import serial
import time

# Configure the serial port (adjust 'COM3' or '/dev/ttyUSB0' as per your setup)
ser = serial.Serial('COM10', 9600, timeout=1)
time.sleep(2)  # Allow time for the connection to establish

try:
    while True:
        if ser.in_waiting > 0:
            gsr_value = ser.readline().decode('utf-8').strip()
            print(f"GSR Value: {gsr_value}")
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
