# GSR Real-Time ML Analysis Integration
# Combines your serial data collection with the ML stress analyzer

import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np
from GSR_ML_analyzer import GSRStressAnalyzer
from datetime import datetime
import pandas as pd

# Enable interactive plotting
plt.ion()

class RealTimeGSRAnalyzer:
    def __init__(self, port='COM10', baud_rate=9600):
        # Serial configuration
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        
        # Initialize ML analyzer
        self.ml_analyzer = GSRStressAnalyzer()
        
        # Data storage for plotting
        self.max_points = 100
        self.gsr_data = deque(maxlen=self.max_points)
        self.time_data = deque(maxlen=self.max_points)
        self.stress_levels = deque(maxlen=self.max_points)
        self.start_time = time.time()
        
        # Analysis results storage
        self.current_stats = None
        self.current_stress = "Normal"
        self.current_confidence = 0.0
        self.recommendations = []
        self.future_prediction = ""
        
        # Initialize connection
        self.connect_serial()
        
        # Setup plot
        self.setup_plot()
        
        # Counter for periodic analysis
        self.analysis_counter = 0
        
    def connect_serial(self):
        """Establish serial connection"""
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)
            print(f"Connected to {self.port} at {self.baud_rate} baud")
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            print("Please check your port and run as administrator if needed")
            self.ser = None
    
    def setup_plot(self):
        """Initialize the comprehensive plotting interface"""
        # Create figure with multiple subplots
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main GSR plot (top)
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
        self.line_gsr, = self.ax_main.plot([], [], 'b-', linewidth=2, label='GSR Signal')
        self.ax_main.set_title('Real-Time GSR Data with ML Analysis', fontsize=16, fontweight='bold')
        self.ax_main.set_xlabel('Time (seconds)')
        self.ax_main.set_ylabel('GSR Value')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend()
        
        # Stress level indicator (bottom left)
        self.ax_stress = plt.subplot2grid((4, 4), (2, 0), colspan=2)
        self.stress_colors = {'Relaxed': 'green', 'Normal': 'orange', 'Stressed': 'red'}
        self.stress_bar = self.ax_stress.bar(['Current Stress'], [1], color='gray', alpha=0.7)
        self.ax_stress.set_title('Stress Level Classification')
        self.ax_stress.set_ylim(0, 1)
        self.ax_stress.set_ylabel('Confidence')
        
        # Statistics display (bottom right)
        self.ax_stats = plt.subplot2grid((4, 4), (2, 2), colspan=2)
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', transform=self.ax_stats.transAxes,
                                           fontsize=10, verticalalignment='top')
        
        # Recommendations display (bottom)
        self.ax_rec = plt.subplot2grid((4, 4), (3, 0), colspan=4)
        self.ax_rec.axis('off')
        self.rec_text = self.ax_rec.text(0.02, 0.8, '', transform=self.ax_rec.transAxes,
                                       fontsize=11, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Current value display on main plot
        self.value_text = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes,
                                          fontsize=12, fontweight='bold',
                                          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Set initial limits
        self.ax_main.set_xlim(0, self.max_points)
        self.ax_main.set_ylim(0, 100)
        
    def read_gsr_data(self):
        """Read GSR data from serial port"""
        if not self.ser or not self.ser.in_waiting:
            return None
        
        try:
            raw_data = self.ser.readline().decode('utf-8').strip()
            if raw_data:
                return float(raw_data)
        except (ValueError, UnicodeDecodeError) as e:
            print(f"Data parsing error: {e}")
        except Exception as e:
            print(f"Serial read error: {e}")
        
        return None
    
    def update_analysis(self, gsr_value):
        """Update ML analysis with new GSR data"""
        # Add data to ML analyzer
        self.ml_analyzer.add_gsr_data(gsr_value)
        
        # Perform periodic comprehensive analysis (every 10 data points)
        self.analysis_counter += 1
        if self.analysis_counter >= 10:
            self.analysis_counter = 0
            
            # Get comprehensive analysis
            self.current_stats, comparison = self.ml_analyzer.get_current_statistics()
            if self.current_stats:
                self.current_stress, self.current_confidence = self.ml_analyzer.classify_stress_level(gsr_value)
                self.recommendations = self.ml_analyzer.get_recommendations(self.current_stress)
                future_stress, self.future_prediction = self.ml_analyzer.predict_future_stress(30)
                
                # Print comprehensive report every 50 data points
                if len(self.ml_analyzer.gsr_history) % 50 == 0:
                    print("\\n" + "="*60)
                    print(self.ml_analyzer.generate_report())
                    print("="*60 + "\\n")
    
    def update_plot(self, frame):
        """Animation update function"""
        gsr_value = self.read_gsr_data()
        
        if gsr_value is not None:
            current_time = time.time() - self.start_time
            
            # Store data
            self.gsr_data.append(gsr_value)
            self.time_data.append(current_time)
            
            # Update ML analysis
            self.update_analysis(gsr_value)
            
            # Store stress level for coloring
            self.stress_levels.append(self.current_stress)
            
            # Update main GSR plot
            self.line_gsr.set_data(list(self.time_data), list(self.gsr_data))
            
            # Auto-adjust main plot axes
            if len(self.gsr_data) > 10:
                self.ax_main.set_xlim(current_time - self.max_points, current_time + 5)
                
                data_array = np.array(list(self.gsr_data))
                y_min, y_max = max(0, data_array.min() - 10), data_array.max() + 10
                self.ax_main.set_ylim(y_min, y_max)
            
            # Update current value display
            self.value_text.set_text(f'Current GSR: {gsr_value:.1f}\\nStress: {self.current_stress}')
            
            # Update stress level bar
            if self.current_stress in self.stress_colors:
                color = self.stress_colors[self.current_stress]
                self.stress_bar[0].set_color(color)
                self.stress_bar[0].set_height(self.current_confidence)
            
            # Update statistics display
            if self.current_stats:
                stats_info = f"""ML ANALYSIS RESULTS:
                
Current Statistics:
• Mean GSR: {self.current_stats['mean']:.1f}
• Std Dev: {self.current_stats['std']:.1f}
• Range: {self.current_stats['range']:.1f}
• Samples: {self.current_stats['samples']}

Classification:
• Stress Level: {self.current_stress}
• Confidence: {self.current_confidence:.1%}

Prediction:
• {self.future_prediction}"""
                
                self.stats_text.set_text(stats_info)
            
            # Update recommendations display
            if self.recommendations:
                rec_info = f"💡 PERSONALIZED RECOMMENDATIONS:\\n"
                for i, rec in enumerate(self.recommendations, 1):
                    rec_info += f"{i}. {rec}\\n"
                
                self.rec_text.set_text(rec_info)
            
            # Console output (keeping original functionality)
            print(f"GSR: {gsr_value:.1f} | Stress: {self.current_stress} | Confidence: {self.current_confidence:.1%}")
        
        return (self.line_gsr, self.value_text, self.stats_text, 
                self.rec_text, self.stress_bar[0])
    
    def start_monitoring(self):
        """Start the real-time monitoring with ML analysis"""
        if not self.ser:
            print("No serial connection available!")
            return
        
        print("Starting Real-Time GSR ML Analysis...")
        print("Close the plot window to stop monitoring.")
        
        # Create animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50,
                               blit=False, cache_frame_data=False)
        
        # Setup cleanup on window close
        def cleanup_on_close(event):
            self.cleanup()
        
        self.fig.canvas.mpl_connect('close_event', cleanup_on_close)
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and save session data"""
        if self.ser:
            try:
                self.ser.close()
                print("Serial connection closed")
            except:
                pass
        
        # Save session data
        if len(self.ml_analyzer.gsr_history) > 0:
            session_data = {
                'timestamp': [d['timestamp'] for d in self.ml_analyzer.gsr_history],
                'gsr_value': [d['value'] for d in self.ml_analyzer.gsr_history]
            }
            
            df = pd.DataFrame(session_data)
            filename = f"gsr_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"Session data saved to {filename}")
        
        # Save the trained model
        self.ml_analyzer.save_model()
        
        # Generate final report
        print("\\n" + "="*60)
        print("FINAL SESSION REPORT:")
        print(self.ml_analyzer.generate_report())
        print("="*60)

# Usage example
if __name__ == "__main__":
    # Create the real-time analyzer
    analyzer = RealTimeGSRAnalyzer(port='COM10', baud_rate=9600)
    
    # Start monitoring
    analyzer.start_monitoring()