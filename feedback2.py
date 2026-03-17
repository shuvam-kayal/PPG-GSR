import sys
import serial
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, lfilter, filtfilt
from collections import deque
import threading
from queue import Queue, Empty
import time

# --- Hardware Configuration ---
SERIAL_PORT = 'COM3'  # TODO: Change this to your actual Arduino port
BAUD_RATE = 115200
FS = 50.0              # Increased to 50Hz for better resolution
WINDOW_SIZE = 500      # 10 seconds at 50Hz
DISPLAY_REFRESH_MS = 33 # ~30 FPS for smoother animation

# --- Optimized DSP Pipeline ---
class RealTimeFilter:
    """Optimized filter with minimal latency"""
    def __init__(self, lowcut=0.1, highcut=0.8, fs=FS, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Use direct-form II transposed for better numerical stability
        self.b, self.a = butter(order, [low, high], btype='band', output='ba')
        
        # Initialize filter states
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
        
        # Pre-compute for speed
        self.a0 = self.a[0]
        self.a = self.a / self.a0
        self.b = self.b / self.a0
        
    def filter_sample(self, sample):
        """Process a single sample with minimal overhead"""
        # Direct form II transposed implementation (faster than sosfilt for single samples)
        y = self.b[0] * sample + self.zi[0]
        
        # Update state
        for i in range(len(self.zi) - 1):
            self.zi[i] = self.b[i + 1] * sample + self.zi[i + 1] - self.a[i + 1] * y
        if len(self.zi) > 0:
            self.zi[-1] = self.b[-1] * sample - self.a[-1] * y
            
        return y

# --- Optimized Data Buffer with Circular Buffer ---
class CircularBuffer:
    """Lock-free circular buffer for maximum performance"""
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size)
        self.index = 0
        self.is_filled = False
        
    def add(self, value):
        self.buffer[self.index] = value
        self.index += 1
        if self.index >= self.size:
            self.index = 0
            self.is_filled = True
            
    def get_ordered(self):
        """Return buffer in chronological order"""
        if not self.is_filled:
            return self.buffer[:self.index]
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))
    
    def get_last_n(self, n):
        """Get last N samples efficiently"""
        if not self.is_filled:
            if self.index < n:
                return self.buffer[:self.index]
            return self.buffer[self.index-n:self.index]
        
        if n >= self.size:
            return self.get_ordered()
            
        start = self.index - n
        if start >= 0:
            return self.buffer[start:self.index]
        else:
            return np.concatenate((self.buffer[start:], self.buffer[:self.index]))

# --- Optimized Serial Reader Thread ---
class SerialReader(QtCore.QObject):
    data_received = QtCore.pyqtSignal(float)
    
    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = True
        self.serial = None
        self.buffer = bytearray()
        
    def start(self):
        """Start the reader thread"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.001,  # Minimal timeout for low latency
                write_timeout=0.001
            )
            # Flush buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Start reading thread
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            print(f"Serial connection failed: {e}")
            sys.exit(1)
            
    def _read_loop(self):
        """High-performance serial reading loop"""
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    # Read all available data at once
                    data = self.serial.read(self.serial.in_waiting)
                    self.buffer.extend(data)
                    
                    # Process complete lines
                    while b'\n' in self.buffer:
                        line, self.buffer = self.buffer.split(b'\n', 1)
                        try:
                            line_str = line.decode('utf-8').strip()
                            if "IR_Value:" in line_str:
                                # Fast parsing - avoid regex/split overhead
                                ir_start = line_str.find(':') + 1
                                ir_end = line_str.find(',', ir_start)
                                if ir_end == -1:
                                    continue
                                    
                                ir_str = line_str[ir_start:ir_end].strip()
                                ir_val = float(ir_str)
                                
                                # Emit via signal (thread-safe)
                                self.data_received.emit(ir_val)
                                
                        except (ValueError, IndexError):
                            pass
                            
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.001)
                
    def stop(self):
        self.running = False
        if self.serial:
            self.serial.close()

# --- Optimized Real-Time GUI ---
class RespiratoryMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_processing()
        self.init_serial()
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.perf_counter()
        
    def init_ui(self):
        """Initialize optimized UI"""
        self.setWindowTitle("Medical-Grade Respiratory Monitor")
        self.setGeometry(100, 100, 1000, 600)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('k')
        layout.addWidget(self.plot_widget)
        
        # Create plot
        self.plot = self.plot_widget.addPlot(title="Real-Time Respiratory Waveform")
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'Time (Samples)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Use faster line drawing
        self.curve = self.plot.plot(pen=pg.mkPen('c', width=2), 
                                    antialias=False,  # Disable for speed
                                    clipToView=True)  # Optimize rendering
        
        # Status label
        self.feedback_label = QtWidgets.QLabel("Calibrating...")
        self.feedback_label.setStyleSheet("color: #cccccc; font-size: 22pt; font-weight: bold;")
        self.feedback_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.feedback_label)
        
        # Performance label (for debugging)
        self.perf_label = QtWidgets.QLabel("")
        self.perf_label.setStyleSheet("color: #666666; font-size: 10pt;")
        self.perf_label.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(self.perf_label)
        
    def init_processing(self):
        """Initialize processing pipeline"""
        # Create filter
        self.filter = RealTimeFilter(lowcut=0.1, highcut=0.8, fs=FS)
        
        # Data buffers
        self.raw_buffer = CircularBuffer(WINDOW_SIZE)
        self.filtered_buffer = CircularBuffer(WINDOW_SIZE)
        
        # Calibration
        self.calibration_samples = []
        self.baseline_dc = 0.0
        self.is_calibrated = False
        self.calibration_needed = 50
        
        # Signal quality metrics
        self.snr_estimate = 0.0
        self.breath_rate = 0.0
        
        # Adaptive thresholds
        self.peak_tracker = deque(maxlen=10)
        self.valley_tracker = deque(maxlen=10)
        self.amplitude_history = deque(maxlen=5)
        
        # Timing
        self.sample_count = 0
        self.last_breath_time = time.perf_counter()
        
    def init_serial(self):
        """Initialize serial reader"""
        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE)
        self.serial_reader.data_received.connect(self.process_sample)
        self.serial_reader.start()
        
        # Update timer (UI refresh)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(DISPLAY_REFRESH_MS)
        
    @QtCore.pyqtSlot(float)
    def process_sample(self, ir_val):
        """Process incoming sample (called from serial thread)"""
        # Store raw
        self.raw_buffer.add(ir_val)
        
        if not self.is_calibrated:
            self.calibration_samples.append(ir_val)
            if len(self.calibration_samples) >= self.calibration_needed:
                # Calculate robust baseline (median instead of mean for outlier rejection)
                self.baseline_dc = np.median(self.calibration_samples)
                self.is_calibrated = True
                self.feedback_label.setText("Calibrated ✓")
                self.feedback_label.setStyleSheet("color: #00ff00; font-size: 22pt; font-weight: bold;")
                print("Calibration complete")
            return
            
        # Center and filter
        centered = ir_val - self.baseline_dc
        filtered = self.filter.filter_sample(centered)
        
        # Store filtered
        self.filtered_buffer.add(filtered)
        self.sample_count += 1
        
        # Real-time analysis (every 10 samples for efficiency)
        if self.sample_count % 10 == 0:
            self.analyze_breathing()
            
    def analyze_breathing(self):
        """Optimized breathing analysis"""
        # Get recent window
        recent = self.filtered_buffer.get_last_n(30)  # Last 0.6 seconds at 50Hz
        
        if len(recent) < 10:
            return
            
        # Track peaks and valleys
        if len(recent) > 2:
            # Simple peak detection
            if recent[-2] > recent[-3] and recent[-2] > recent[-1]:
                self.peak_tracker.append(recent[-2])
            # Simple valley detection
            if recent[-2] < recent[-3] and recent[-2] < recent[-1]:
                self.valley_tracker.append(recent[-2])
        
        # Calculate current amplitude
        if self.peak_tracker and self.valley_tracker:
            current_amplitude = np.mean(self.peak_tracker) - np.mean(self.valley_tracker)
            self.amplitude_history.append(current_amplitude)
            
        # Estimate breath rate using zero crossings
        if len(recent) > 5:
            zero_crossings = np.where(np.diff(np.signbit(recent)))[0]
            if len(zero_crossings) > 1:
                # Time between zero crossings ~ half breath period
                avg_period = np.mean(np.diff(zero_crossings)) * 2 * (1000/FS)  # in ms
                if avg_period > 1000:  # Sanity check (>1 second period)
                    self.breath_rate = 60 / (avg_period / 1000)
        
    def update_display(self):
        """Update UI (called from main thread)"""
        if not self.is_calibrated:
            # Show calibration progress
            progress = len(self.calibration_samples) / self.calibration_needed * 100
            self.feedback_label.setText(f"Calibrating... {progress:.0f}%")
            return
            
        # Get ordered data for display
        display_data = self.filtered_buffer.get_ordered()
        
        if len(display_data) > 10:
            # Update plot
            self.curve.setData(display_data)
            
            # Auto-range Y axis occasionally
            if self.sample_count % 100 == 0:
                data_range = np.ptp(display_data)  # peak-to-peak
                if data_range > 0:
                    mean_val = np.mean(display_data)
                    self.plot.setYRange(mean_val - data_range, mean_val + data_range)
            
            # Update feedback with optimized logic
            self.update_feedback(display_data)
            
        # Performance monitoring
        now = time.perf_counter()
        frame_time = (now - self.last_frame_time) * 1000  # ms
        self.frame_times.append(frame_time)
        self.last_frame_time = now
        
        if len(self.frame_times) > 10:
            avg_frame_time = np.mean(self.frame_times)
            self.perf_label.setText(f"Frame: {avg_frame_time:.1f}ms | Rate: {self.breath_rate:.1f} BPM")
        
    def update_feedback(self, data):
        """Intelligent breathing feedback"""
        if len(data) < 50:
            return
            
        # Get windows
        recent = data[-40:]  # Last ~0.8 seconds
        full = data
        
        # Calculate metrics
        recent_amp = np.ptp(recent)
        full_amp = np.ptp(full)
        
        # Use historical amplitude for better threshold
        if self.amplitude_history:
            normal_amp = np.mean(self.amplitude_history)
        else:
            normal_amp = full_amp
            
        # Apnea detection
        if recent_amp < (normal_amp * 0.25) or recent_amp < 2.0:
            self.feedback_label.setText("⚠ APNEA DETECTED ⚠")
            self.feedback_label.setStyleSheet("color: #ff0000; font-size: 22pt; font-weight: bold;")
            return
            
        # Breath phase detection with hysteresis
        recent_mean = np.mean(recent[-10:])
        prev_mean = np.mean(recent[-20:-10])
        
        # Add hysteresis to prevent flickering
        threshold = normal_amp * 0.05
        
        if recent_mean < prev_mean - threshold:
            self.feedback_label.setText("INHALING ↑")
            self.feedback_label.setStyleSheet("color: #00ffff; font-size: 22pt; font-weight: bold;")
        elif recent_mean > prev_mean + threshold:
            self.feedback_label.setText("EXHALING ↓")
            self.feedback_label.setStyleSheet("color: #ffaa00; font-size: 22pt; font-weight: bold;")
        # Else keep previous state (hysteresis)
        
    def closeEvent(self, event):
        """Clean shutdown"""
        self.serial_reader.stop()
        event.accept()

# --- Main Application ---
def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set high DPI scaling for better performance
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    
    monitor = RespiratoryMonitor()
    monitor.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()