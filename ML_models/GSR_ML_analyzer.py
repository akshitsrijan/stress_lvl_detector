# GSR Stress Analysis ML Model - CORRECTED VERSION
# Fixed TypeError in _extract_single_features method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
import joblib
import warnings
from collections import deque
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class GSRStressAnalyzer:
    def __init__(self):
        # Healthy baseline values (adjust based on your sensor calibration)
        self.healthy_baseline = {
            'mean_gsr': 18.5,
            'std_gsr': 4.2,
            'min_gsr': 8.0,
            'max_gsr': 35.0,
            'variability': 3.8
        }
        
        # Stress classification thresholds
        self.stress_thresholds = {
            'relaxed_max': 15,
            'normal_max': 35,
            'stressed_min': 35
        }
        
        # Feature history for predictions
        self.gsr_history = deque(maxlen=100)
        self.feature_history = deque(maxlen=50)
        self.stress_history = deque(maxlen=50)
        
        # ML Models
        self.classifier = None
        self.predictor = None
        self.scaler = StandardScaler()
        
        # Recommendation database
        self.recommendations = {
            'Relaxed': [
                "Maintain your current calm state with light meditation",
                "Consider gentle stretching or yoga",
                "Good time for creative activities",
                "Keep up your current routine - you're doing great!"
            ],
            'Normal': [
                "Take regular short breaks every 30 minutes",
                "Stay hydrated and maintain good posture",
                "Consider brief mindfulness exercises",
                "Monitor your stress patterns throughout the day"
            ],
            'Stressed': [
                "Try 4-7-8 breathing: Inhale 4s, hold 7s, exhale 8s",
                "Take a 5-10 minute walk outside",
                "Listen to calming music or nature sounds",
                "Practice progressive muscle relaxation",
                "Consider reducing caffeine intake"
            ]
        }
        
        print("GSR Stress Analyzer initialized!")
        self._create_synthetic_training_data()
        self._train_models()
    
    def _create_synthetic_training_data(self):
        """Create synthetic training data for initial model training"""
        np.random.seed(42)
        
        # Generate synthetic GSR data for different stress states
        n_samples = 300
        
        # Relaxed state (lower GSR, low variability)
        relaxed_gsr = np.random.normal(12, 3, n_samples//3)
        relaxed_features = self._extract_features_batch(relaxed_gsr)
        relaxed_labels = ['Relaxed'] * len(relaxed_features)
        
        # Normal state (moderate GSR, moderate variability)
        normal_gsr = np.random.normal(25, 5, n_samples//3)
        normal_features = self._extract_features_batch(normal_gsr)
        normal_labels = ['Normal'] * len(normal_features)
        
        # Stressed state (higher GSR, high variability)
        stressed_gsr = np.random.normal(45, 8, n_samples//3)
        stressed_features = self._extract_features_batch(stressed_gsr)
        stressed_labels = ['Stressed'] * len(stressed_features)
        
        # Combine all data
        self.training_features = np.vstack([relaxed_features, normal_features, stressed_features])
        self.training_labels = relaxed_labels + normal_labels + stressed_labels
        
        print(f"Synthetic training data created: {len(self.training_labels)} samples")
    
    def _extract_features_batch(self, gsr_values):
        """Extract features from batch of GSR values"""
        features = []
        window_size = 10
        
        for i in range(window_size, len(gsr_values)):
            window = gsr_values[i-window_size:i]
            feature_vector = self._extract_single_features(window, gsr_values[i])
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_single_features(self, window, current_value):
        """Extract features from a single GSR window - FIXED VERSION"""
        # Convert window to numpy array to enable boolean indexing
        window_array = np.array(window)
        
        # Handle edge cases
        if len(window_array) == 0:
            return np.array([current_value, current_value, 0, 0, 0, 0, 0])
        
        # Calculate mean once for efficiency
        window_mean = np.mean(window_array)
        window_std = np.std(window_array) if len(window_array) > 1 else 0
        window_range = np.max(window_array) - np.min(window_array) if len(window_array) > 1 else 0
        
        # Count values above average (FIXED: using numpy array for boolean indexing)
        above_avg_count = np.sum(window_array > window_mean)
        
        # Calculate trend (gradient)
        trend = np.gradient(window_array)[-1] if len(window_array) > 1 else 0
        
        return np.array([
            current_value,                    # Current GSR value
            window_mean,                      # Mean of recent values
            window_std,                       # Standard deviation
            window_range,                     # Range
            current_value - window_mean,      # Deviation from recent mean
            above_avg_count,                  # Above-average count (FIXED)
            trend,                           # Recent trend
        ])
    
    def _train_models(self):
        """Train the ML models"""
        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_features, self.training_labels, 
            test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate classifier
        y_pred = self.classifier.predict(X_test_scaled)
        print("Classifier Performance:")
        print(classification_report(y_test, y_pred))
        
        # Train predictor for future GSR values
        self.predictor = LinearRegression()
        
        print("ML Models trained successfully!")
    
    def add_gsr_data(self, gsr_value):
        """Add new GSR data point and update history"""
        self.gsr_history.append({
            'value': gsr_value,
            'timestamp': datetime.now()
        })
        
        # Extract features if we have enough history
        if len(self.gsr_history) >= 10:
            recent_values = [d['value'] for d in list(self.gsr_history)[-10:]]
            features = self._extract_single_features(recent_values[:-1], gsr_value)
            self.feature_history.append(features)
    
    def get_current_statistics(self):
        """Calculate current session statistics and compare with healthy baseline"""
        if len(self.gsr_history) < 5:
            return None, None
        
        recent_values = [d['value'] for d in self.gsr_history]
        
        # Current statistics
        current_stats = {
            'mean': np.mean(recent_values),
            'median': np.median(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'range': np.max(recent_values) - np.min(recent_values),
            'variability': np.std(np.diff(recent_values)) if len(recent_values) > 1 else 0,
            'samples': len(recent_values)
        }
        
        # Comparison with healthy baseline
        comparison = {
            'mean_diff': current_stats['mean'] - self.healthy_baseline['mean_gsr'],
            'std_diff': current_stats['std'] - self.healthy_baseline['std_gsr'],
            'variability_diff': current_stats['variability'] - self.healthy_baseline['variability'],
            'above_healthy_range': current_stats['mean'] > self.healthy_baseline['max_gsr'],
            'within_healthy_range': (self.healthy_baseline['min_gsr'] <= current_stats['mean'] 
                                   <= self.healthy_baseline['max_gsr'])
        }
        
        return current_stats, comparison
    
    def classify_stress_level(self, gsr_value):
        """Classify current stress level using both rule-based and ML approaches"""
        # Rule-based classification (fallback)
        if gsr_value <= self.stress_thresholds['relaxed_max']:
            rule_based = 'Relaxed'
        elif gsr_value <= self.stress_thresholds['normal_max']:
            rule_based = 'Normal'
        else:
            rule_based = 'Stressed'
        
        # ML-based classification (if we have enough features)
        ml_based = rule_based  # Default to rule-based
        confidence = 0.5
        
        if len(self.feature_history) > 0:
            try:
                latest_features = self.feature_history[-1].reshape(1, -1)
                scaled_features = self.scaler.transform(latest_features)
                
                ml_prediction = self.classifier.predict(scaled_features)[0]
                ml_probabilities = self.classifier.predict_proba(scaled_features)[0]
                
                ml_based = ml_prediction
                confidence = np.max(ml_probabilities)
                
            except Exception as e:
                print(f"ML classification error: {e}")
        
        self.stress_history.append({
            'timestamp': datetime.now(),
            'rule_based': rule_based,
            'ml_based': ml_based,
            'confidence': confidence,
            'gsr_value': gsr_value
        })
        
        return ml_based, confidence
    
    def get_recommendations(self, stress_level):
        """Get personalized recommendations based on stress level"""
        base_recommendations = self.recommendations.get(stress_level, [])
        
        # Add personalized recommendations based on history
        if len(self.stress_history) > 10:
            recent_stress = [s['ml_based'] for s in list(self.stress_history)[-10:]]
            stress_pattern = max(set(recent_stress), key=recent_stress.count)
            
            if stress_pattern == 'Stressed' and stress_level == 'Stressed':
                base_recommendations.append("Consider longer-term stress management techniques")
            elif stress_pattern == 'Normal' and stress_level == 'Relaxed':
                base_recommendations.append("You're improving! Keep up the good work")
        
        return np.random.choice(base_recommendations, size=min(2, len(base_recommendations)), replace=False)
    
    def predict_future_stress(self, minutes_ahead=30):
        """Predict future stress level based on current trends"""
        if len(self.gsr_history) < 20:
            return None, "Insufficient data for prediction"
        
        recent_values = [d['value'] for d in list(self.gsr_history)[-20:]]
        timestamps = [i for i in range(len(recent_values))]
        
        try:
            # Fit linear regression to recent trend
            self.predictor.fit(np.array(timestamps).reshape(-1, 1), recent_values)
            
            # Predict future values
            future_timestamp = len(recent_values) + (minutes_ahead / 5)  # Assuming 5-minute intervals
            future_gsr = self.predictor.predict([[future_timestamp]])[0]
            
            # Classify future stress level
            if future_gsr <= self.stress_thresholds['relaxed_max']:
                future_stress = 'Relaxed'
            elif future_gsr <= self.stress_thresholds['normal_max']:
                future_stress = 'Normal'
            else:
                future_stress = 'Stressed'
            
            # Calculate trend
            current_avg = np.mean(recent_values[-5:])
            trend = "increasing" if future_gsr > current_avg else "decreasing"
            
            prediction_text = f"In {minutes_ahead} minutes: {future_stress} (GSR: {future_gsr:.1f}, trend: {trend})"
            
            return future_stress, prediction_text
            
        except Exception as e:
            return None, f"Prediction error: {e}"
    
    def generate_report(self):
        """Generate comprehensive stress analysis report"""
        if len(self.gsr_history) < 5:
            return "Insufficient data for report generation"
        
        stats, comparison = self.get_current_statistics()
        current_gsr = self.gsr_history[-1]['value']
        stress_level, confidence = self.classify_stress_level(current_gsr)
        recommendations = self.get_recommendations(stress_level)
        future_stress, prediction = self.predict_future_stress()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    GSR STRESS ANALYSIS REPORT                ║
╚══════════════════════════════════════════════════════════════╝

📊 CURRENT STATISTICS:
   • Mean GSR: {stats['mean']:.2f} (Healthy: {self.healthy_baseline['mean_gsr']:.1f})
   • Standard Deviation: {stats['std']:.2f} (Healthy: {self.healthy_baseline['std_gsr']:.1f})
   • Range: {stats['range']:.2f}
   • Samples: {stats['samples']}

📈 COMPARISON WITH HEALTHY BASELINE:
   • Mean Difference: {comparison['mean_diff']:+.2f}
   • Variability Difference: {comparison['variability_diff']:+.2f}
   • Within Healthy Range: {'✓' if comparison['within_healthy_range'] else '✗'}

🎯 CURRENT STRESS CLASSIFICATION:
   • Level: {stress_level}
   • Confidence: {confidence:.1%}
   • Current GSR: {current_gsr:.2f}

💡 PERSONALIZED RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"   {i}. {rec}\n"
        
        report += f"""
🔮 FUTURE PREDICTION:
   • {prediction if prediction else 'Not available'}

📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def save_model(self, filepath='gsr_stress_model.pkl'):
        """Save the trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'predictor': self.predictor,
            'healthy_baseline': self.healthy_baseline,
            'stress_thresholds': self.stress_thresholds
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='gsr_stress_model.pkl'):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.predictor = model_data['predictor']
            self.healthy_baseline = model_data['healthy_baseline']
            self.stress_thresholds = model_data['stress_thresholds']
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Using default model.")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = GSRStressAnalyzer()
    
    # Simulate some GSR data (replace with your real serial data)
    test_gsr_values = [12, 15, 18, 22, 28, 35, 42, 38, 25, 20, 16, 14]
    
    print("Testing with simulated GSR data...")
    print("=" * 60)
    
    for i, gsr_value in enumerate(test_gsr_values):
        print(f"\nStep {i+1}: GSR Value = {gsr_value}")
        
        # Add data point
        analyzer.add_gsr_data(gsr_value)
        
        # Get analysis
        if i >= 4:  # Need at least 5 points for analysis
            stress_level, confidence = analyzer.classify_stress_level(gsr_value)
            print(f"Stress Level: {stress_level} (Confidence: {confidence:.1%})")
            
            recommendations = analyzer.get_recommendations(stress_level)
            print("Recommendations:", recommendations)
            
            if i >= 9:  # Need more points for prediction
                future_stress, prediction = analyzer.predict_future_stress(30)
                print("Future Prediction:", prediction)
    
    # Generate final report
    print("\n" + "=" * 60)
    print("FINAL COMPREHENSIVE REPORT:")
    print(analyzer.generate_report())
    
    # Save the model
    analyzer.save_model()