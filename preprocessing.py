import numpy as np

def preprocess_gsr(data):
    # Convert raw data to numpy array
    data = np.array(data, dtype=float)
    
    # Normalize data (0 to 1 range)
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Apply a simple moving average filter
    window_size = 5
    smoothed_data = np.convolve(normalized_data, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed_data

# Example usage
raw_data = [500, 520, 510, 530, 540, 550, 560, 570, 580, 590]
processed_data = preprocess_gsr(raw_data)
print("Processed GSR Data:", processed_data)
