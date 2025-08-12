# Simple GSR Live Plot - Minimal changes to your original code
# This replaces your while loop with a plotting animation

import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Enable interactive mode
plt.ion()

# Configure the serial port (adjust 'COM10' as per your setup)
ser = serial.Serial('COM10', 9600, timeout=1)
time.sleep(2)  # Allow time for the connection to establish

# Data storage
data = deque(maxlen=50)  # Keep last 50 points
timestamps = deque(maxlen=50)
start_time = time.time()

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'g-', linewidth=2)
ax.set_title('Live GSR Data')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('GSR Value')
ax.grid(True)

def animate(frame):
    global ser, data, timestamps, start_time
    
    try:
        if ser.in_waiting > 0:
            gsr_value = ser.readline().decode('utf-8').strip()
            
            if gsr_value:
                try:
                    value = float(gsr_value)
                    current_time = time.time() - start_time
                    
                    # Add data
                    data.append(value)
                    timestamps.append(current_time)
                    
                    # Update plot
                    line.set_data(list(timestamps), list(data))
                    
                    # Auto-scale
                    if len(data) > 5:
                        ax.set_xlim(min(timestamps), max(timestamps) + 1)
                        ax.set_ylim(min(data) - 20, max(data) + 20)
                    
                    # Keep your original print
                    print(f"GSR Value: {gsr_value}")
                    
                except ValueError:
                    print(f"Invalid data: {gsr_value}")
                    
    except Exception as e:
        print(f"Error: {e}")
    
    return line,

# Start animation
ani = FuncAnimation(fig, animate, interval=100, blit=False)

# Show plot
plt.show()

# Keep running (this replaces your while loop)
try:
    plt.show(block=True)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close() 