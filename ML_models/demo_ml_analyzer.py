# GSR ML Demo - FIXED VERSION
# Test the ML analyzer without hardware - No more array ambiguity errors

from GSR_ML_analyzer import GSRStressAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_stress_session():
    """Simulate a realistic GSR stress session"""
    # Simulate different stress phases
    np.random.seed(42)
    
    # Phase 1: Relaxed (5 minutes)
    relaxed_phase = np.random.normal(12, 2, 30) + np.random.normal(0, 0.5, 30)
    
    # Phase 2: Gradual stress increase (10 minutes) 
    stress_buildup = np.linspace(12, 40, 60) + np.random.normal(0, 3, 60)
    
    # Phase 3: High stress (5 minutes)
    stressed_phase = np.random.normal(45, 6, 30) + np.random.normal(0, 2, 30)
    
    # Phase 4: Recovery (10 minutes)
    recovery_phase = np.linspace(45, 18, 60) + np.random.normal(0, 4, 60)
    
    # Combine all phases
    full_session = np.concatenate([relaxed_phase, stress_buildup, 
                                  stressed_phase, recovery_phase])
    
    # Ensure no negative values
    full_session = np.maximum(full_session, 5)
    
    return full_session, ['Relaxed']*30 + ['Building']*60 + ['Stressed']*30 + ['Recovery']*60

def run_comprehensive_demo():
    """Run a comprehensive demonstration of all ML features"""
    print("🚀 GSR ML Stress Analyzer - Comprehensive Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GSRStressAnalyzer()
    
    # Generate realistic test data
    gsr_data, phase_labels = simulate_stress_session()
    
    print(f"📊 Simulating {len(gsr_data)} GSR readings over a stress session...")
    print("Phases: Relaxed → Stress Building → Highly Stressed → Recovery\\n")
    
    # Process data and collect results
    results = []
    
    for i, gsr_value in enumerate(gsr_data):
        # Add data to analyzer
        analyzer.add_gsr_data(gsr_value)
        
        # Perform analysis every 10 points
        if i % 10 == 0 and i > 20:
            stats, comparison = analyzer.get_current_statistics()
            stress_level, confidence = analyzer.classify_stress_level(gsr_value)
            recommendations = analyzer.get_recommendations(stress_level)
            
            if i > 40:  # Predictions need more data
                future_stress, prediction = analyzer.predict_future_stress(30)
            else:
                future_stress, prediction = None, "Insufficient data"
            
            # FIXED: Handle numpy array properly
            rec_text = "No recommendation"
            if len(recommendations) > 0:
                rec_text = recommendations[0] if isinstance(recommendations[0], str) else str(recommendations[0])
            
            results.append({
                'sample': i,
                'phase': phase_labels[i],
                'gsr': gsr_value,
                'predicted_stress': stress_level,
                'confidence': confidence,
                'mean_gsr': stats['mean'] if stats else 0,
                'recommendation': rec_text,
                'prediction': prediction
            })
            
            # Print periodic updates
            if i % 30 == 0:
                print(f"📈 Sample {i}: GSR={gsr_value:.1f}, Phase={phase_labels[i]}, "
                      f"Predicted={stress_level} ({confidence:.1%})")
    
    # Generate final comprehensive report
    print("\\n" + "=" * 60)
    print("📋 FINAL COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    print(analyzer.generate_report())
    
    # Show detailed results table
    print("\\n📊 DETAILED SESSION ANALYSIS:")
    print("-" * 100)
    print(f"{'Sample':<8} {'Phase':<10} {'GSR':<8} {'Predicted':<10} {'Confidence':<12} {'Recommendation':<50}")
    print("-" * 100)
    
    for result in results[-10:]:  # Show last 10 results
        print(f"{result['sample']:<8} {result['phase']:<10} {result['gsr']:<8.1f} "
              f"{result['predicted_stress']:<10} {result['confidence']:<12.1%} "
              f"{result['recommendation'][:47]:<50}")
    
    # Create visualization
    create_demo_visualization(gsr_data, results, phase_labels)
    
    # Test model persistence
    print("\\n💾 Testing model save/load functionality...")
    analyzer.save_model('demo_model.pkl')
    
    # Create new analyzer and load model
    new_analyzer = GSRStressAnalyzer()
    new_analyzer.load_model('demo_model.pkl')
    print("✅ Model successfully saved and loaded!")
    
    return analyzer, results

def create_demo_visualization(gsr_data, results, phase_labels):
    """Create comprehensive visualization of demo results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GSR ML Analysis Demo Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Raw GSR data with phase coloring
    ax1 = axes[0, 0]
    phase_colors = {'Relaxed': 'green', 'Building': 'orange', 
                   'Stressed': 'red', 'Recovery': 'blue'}
    
    for i, (gsr, phase) in enumerate(zip(gsr_data, phase_labels)):
        ax1.scatter(i, gsr, c=phase_colors.get(phase, 'gray'), alpha=0.6, s=20)
    
    ax1.plot(gsr_data, 'k-', alpha=0.3, linewidth=1)
    ax1.set_title('GSR Data by Stress Phase')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('GSR Value')
    ax1.grid(True, alpha=0.3)
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=color, markersize=8, label=phase)
              for phase, color in phase_colors.items()]
    ax1.legend(handles=handles, loc='upper right')
    
    # Plot 2: ML Predictions vs Actual Phases
    ax2 = axes[0, 1]
    if results:
        samples = [r['sample'] for r in results]
        predictions = [r['predicted_stress'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        pred_colors = {'Relaxed': 'green', 'Normal': 'orange', 'Stressed': 'red'}
        colors = [pred_colors.get(pred, 'gray') for pred in predictions]
        
        scatter = ax2.scatter(samples, range(len(samples)), c=colors, 
                             s=[c*100 for c in confidences], alpha=0.7)
        ax2.set_title('ML Stress Predictions\\n(Size = Confidence)')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Prediction Order')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistics over time
    ax3 = axes[1, 0]
    if results:
        samples = [r['sample'] for r in results]
        mean_gsr = [r['mean_gsr'] for r in results]
        
        ax3.plot(samples, mean_gsr, 'b-o', linewidth=2, markersize=4)
        ax3.axhline(y=18.5, color='g', linestyle='--', alpha=0.7, label='Healthy Baseline')
        ax3.set_title('Mean GSR Trend')
        ax3.set_xlabel('Sample Number')
        ax3.set_ylabel('Mean GSR Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confidence levels
    ax4 = axes[1, 1]
    if results:
        confidences = [r['confidence'] for r in results]
        ax4.hist(confidences, bins=10, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.1%}')
        ax4.set_title('Prediction Confidence Distribution')
        ax4.set_xlabel('Confidence Level')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 Visualization created! Close the plot window to continue...")

def quick_demo():
    """Quick demonstration for impatient users"""
    print("⚡ Quick GSR ML Demo")
    print("-" * 30)
    
    analyzer = GSRStressAnalyzer()
    
    # Test with a few sample values
    test_values = [10, 15, 25, 35, 45, 40, 30, 20, 12]
    
    for i, gsr in enumerate(test_values):
        analyzer.add_gsr_data(gsr)
        
        if i >= 4:  # Need some history
            stress, confidence = analyzer.classify_stress_level(gsr)
            recommendations = analyzer.get_recommendations(stress)
            
            # FIXED: Handle numpy array properly
            rec_text = "No recommendation available"
            if len(recommendations) > 0:
                rec_text = str(recommendations[0])[:40] + "..."
            
            print(f"GSR: {gsr:2d} → {stress:8s} ({confidence:5.1%}) | {rec_text}")
    
    print("\\n✅ Quick demo complete!")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Comprehensive Demo (5 minutes)")
    print("2. Quick Demo (30 seconds)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            analyzer, results = run_comprehensive_demo()
        except Exception as e:
            print(f"Error in comprehensive demo: {e}")
            print("Running quick demo instead...")
            quick_demo()
    else:
        quick_demo()
    
    print("\\n🎉 Demo complete! Ready to use with real GSR hardware.")
    print("\\nNext steps:")
    print("1. Connect your GSR sensor")
    print("2. Update COM port in realtime_gsr_ml.py")
    print("3. Run: python realtime_gsr_ml.py")