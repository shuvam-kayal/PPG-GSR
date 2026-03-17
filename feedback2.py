import sys
import serial
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, sosfilt, sosfilt_zi

# --- Hardware Configuration ---
SERIAL_PORT = 'COM3'  # TODO: Change this to your actual Arduino port
BAUD_RATE = 115200
FS = 20.0             # Expected effective sample rate
WINDOW_SIZE = 200     # 10 seconds of data at 20 Hz

# --- DSP Pipeline: Stateful Real-Time Filter ---
def init_filter(lowcut=0.15, highcut=0.5, fs=FS, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    zi = sosfilt_zi(sos) 
    return sos, zi

sos, zi_state = init_filter()

# --- Real-Time GUI Setup ---
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="Medical-Grade Respiratory Monitor")
win.resize(1000, 600)
win.setBackground('k')

plot = win.addPlot(title="Real-Time Respiratory Waveform (Auto-Scaling)")
plot.setLabel('left', 'Amplitude Modulations')
plot.setLabel('bottom', 'Time (Samples)')
plot.showGrid(x=True, y=True)

# Give the auto-range a baseline to prevent "invisible line" glitch on 0.0 arrays
plot.setYRange(-10, 10)
plot.enableAutoRange(axis='y', enable=True) 

curve = plot.plot(pen=pg.mkPen('c', width=3)) 

win.nextRow() # Moves to the space below the plot
feedback_label = win.addLabel("Initializing...", color='#cccccc', size='22pt', bold=True)

# --- Data Buffers ---
filtered_data = np.zeros(WINDOW_SIZE)

# --- Serial Port Initialization ---
try:
    # CRITICAL FIX 1: Increased timeout to 0.1s so it NEVER chops a line in half
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {SERIAL_PORT}. Please place your finger on the sensor.")
except Exception as e:
    print(f"Failed to connect to serial port: {e}")
    sys.exit(1)

# --- Calibration Variables ---
calibration_samples = []
is_calibrated = False
baseline_dc = 0.0

# --- Main Update Loop ---
def update():
    global filtered_data, zi_state, is_calibrated, baseline_dc, calibration_samples
    
    new_data = False

    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            
            if not line:
                continue
            
            if "IR_Value:" in line and "Red_Value:" in line:
                parts = line.split(',')
                ir_str = parts[0].split(':')[1].strip()
                ir_val = float(ir_str)
                
                if not is_calibrated:
                    calibration_samples.append(ir_val)
                    if len(calibration_samples) > 50:
                        baseline_dc = np.mean(calibration_samples)
                        
                        # Reset the math filter's memory to 0
                        zi_state = zi_state * 0  
                        
                        is_calibrated = True
                        feedback_label.setText("Calibrated. Waiting for breath...", color='#00ff00')
                        print("Calibrated! Plotting live respiratory wave...")
                    continue

                # Subtract the baseline DC to keep the signal centered at 0
                centered_sample = ir_val - baseline_dc
                
                # Process exactly ONE sample using the historical state
                filtered_sample, zi_state = sosfilt(sos, [centered_sample], zi=zi_state)
                
                # Update the visual plotting buffer
                filtered_data[:-1] = filtered_data[1:]
                
                # Safely extract the exact float value using [0]
                filtered_data[-1] = filtered_sample[0] 
                new_data = True
                
        except (ValueError, IndexError, UnicodeDecodeError) as e:
            # Let's print the error so it can never fail silently again!
            print(f"Ignored a bad data read: {e}")
            pass

    if new_data and is_calibrated:
        # Clean up any potential math errors before rendering to prevent freezing
        valid_data = np.nan_to_num(filtered_data, nan=0.0, posinf=0.0, neginf=0.0)
        curve.setData(valid_data)

        # --- NEW: Feedback Logic ---
        # 1. Apnea (Stopped Breathing) Detection
        recent_window = valid_data[-50:] # The last 2.5 seconds
        full_window = valid_data         # The full 10 seconds
        
        recent_amplitude = np.max(recent_window) - np.min(recent_window)
        full_amplitude = np.max(full_window) - np.min(full_window)

        # If the recent wave shrinks to less than 30% of your normal breathing wave
        # (or drops below a baseline minimum of 5.0 to prevent division glitches)
        if recent_amplitude < (full_amplitude * 0.3) or recent_amplitude < 5.0:  
            feedback_label.setText("Stopped Breathing", color='#ff0000') # Red
        else:
            # 2. Phase Detection (Inhale vs Exhale)
            # Use a small smoothing window (e.g., 5 samples / 250ms) to detect the slope
            current_mean = np.mean(valid_data[-5:])
            prev_mean = np.mean(valid_data[-10:-5])

            if current_mean < prev_mean:
                feedback_label.setText("Inhaling ↑", color='#00ffff') # Cyan
            else:
                feedback_label.setText("Exhaling ↓", color='#ffaa00') # Orange

# CRITICAL FIX 3: Poll the GUI faster than Arduino sends data (20ms vs 50ms)
# This prevents the serial buffer from overflowing and losing sync
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20) 

if __name__ == '__main__':
    sys.exit(app.exec_())