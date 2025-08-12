// Global variables
let sessionData = [];
let gsrChart = null;
let pieChart = null;
let currentTheme = 'light';

// Sample data from the provided JSON
const sampleData = [
    {"timestamp":"2025-08-12T10:00:00","gsr":12},
    {"timestamp":"2025-08-12T10:01:00","gsr":25},
    {"timestamp":"2025-08-12T10:02:00","gsr":40},
    {"timestamp":"2025-08-12T10:03:00","gsr":22},
    {"timestamp":"2025-08-12T10:04:00","gsr":18},
    {"timestamp":"2025-08-12T10:05:00","gsr":31}
];

// Stress classification thresholds
const STRESS_THRESHOLDS = {
    RELAXED: 15,
    STRESSED: 35
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    loadSampleData(); // Auto-load sample data
    
    // Initialize forecast display
    document.getElementById('predictedGsr').textContent = 'Click Predict';
    document.getElementById('predictedStress').textContent = 'Click Predict';
});

// Event Listeners
function setupEventListeners() {
    // File upload
    const fileInput = document.getElementById('csvUpload');
    fileInput.addEventListener('change', handleFileUpload);
    
    // Sample data button
    document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
    
    // Prediction button
    document.getElementById('predictBtn').addEventListener('click', handlePrediction);
    
    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
}

// Parse CSV data
function parseSessionCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];
    
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values.length >= 2 && values[1].trim() !== '') {
            const gsrValue = parseFloat(values[1].trim());
            if (!isNaN(gsrValue)) {
                data.push({
                    timestamp: values[0].trim(),
                    gsr: gsrValue
                });
            }
        }
    }
    
    return data;
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showUploadStatus('Please select a CSV file', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const csvData = parseSessionCSV(e.target.result);
            if (csvData.length === 0) {
                showUploadStatus('No valid data found in CSV file', 'error');
                return;
            }
            
            sessionData = csvData;
            updateDashboard();
            showUploadStatus(`Successfully loaded ${csvData.length} data points`, 'success');
        } catch (error) {
            console.error('CSV parsing error:', error);
            showUploadStatus('Error parsing CSV file', 'error');
        }
    };
    reader.onerror = function() {
        showUploadStatus('Error reading file', 'error');
    };
    reader.readAsText(file);
}

// Load sample data
function loadSampleData() {
    sessionData = [...sampleData];
    updateDashboard();
    showUploadStatus(`Sample data loaded (${sampleData.length} points)`, 'success');
}

// Show upload status
function showUploadStatus(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = message;
    statusDiv.className = `upload-status ${type}`;
    
    // Clear status after 5 seconds
    setTimeout(() => {
        statusDiv.textContent = '';
        statusDiv.className = 'upload-status';
    }, 5000);
}

// Compute statistics
function computeStats(data) {
    if (data.length === 0) return null;
    
    const values = data.map(d => d.gsr);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return {
        mean: mean.toFixed(2),
        stdDev: stdDev.toFixed(2),
        min: min.toFixed(2),
        max: max.toFixed(2),
        range: (max - min).toFixed(2),
        count: values.length
    };
}

// Classify stress level
function classifyStress(gsrValue) {
    if (gsrValue < STRESS_THRESHOLDS.RELAXED) {
        return { level: 'Relaxed', class: 'relaxed' };
    } else if (gsrValue <= STRESS_THRESHOLDS.STRESSED) {
        return { level: 'Normal', class: 'normal' };
    } else {
        return { level: 'Stressed', class: 'stressed' };
    }
}

// Count stress classifications
function countStressLevels(data) {
    const counts = { relaxed: 0, normal: 0, stressed: 0 };
    
    data.forEach(point => {
        const classification = classifyStress(point.gsr);
        counts[classification.class]++;
    });
    
    return counts;
}

// Initialize charts
function initializeCharts() {
    // GSR Line Chart
    const gsrCtx = document.getElementById('gsrChart').getContext('2d');
    gsrChart = new Chart(gsrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'GSR Signal',
                    data: [],
                    borderColor: '#1FB8CD',
                    backgroundColor: 'rgba(31, 184, 205, 0.1)',
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'Stress Threshold',
                    data: [],
                    borderColor: 'rgba(255, 84, 89, 0.6)',
                    backgroundColor: 'rgba(255, 84, 89, 0.1)',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'GSR (µS)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Sample Index'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });

    // Stress Pie Chart
    const pieCtx = document.getElementById('stressPieChart').getContext('2d');
    pieChart = new Chart(pieCtx, {
        type: 'pie',
        data: {
            labels: ['Relaxed', 'Normal', 'Stressed'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Update dashboard with current data
function updateDashboard() {
    if (sessionData.length === 0) return;
    
    // Update statistics
    updateStatistics();
    
    // Update charts
    updateGSRChart();
    updatePieChart();
}

// Update statistics display
function updateStatistics() {
    const stats = computeStats(sessionData);
    if (!stats) return;
    
    document.getElementById('meanGsr').textContent = `${stats.mean} µS`;
    document.getElementById('stdGsr').textContent = `${stats.stdDev} µS`;
    document.getElementById('minGsr').textContent = `${stats.min} µS`;
    document.getElementById('maxGsr').textContent = `${stats.max} µS`;
    document.getElementById('rangeGsr').textContent = `${stats.range} µS`;
    document.getElementById('sampleCount').textContent = stats.count;
}

// Update GSR line chart
function updateGSRChart() {
    const labels = sessionData.map((_, index) => index + 1);
    const gsrValues = sessionData.map(d => d.gsr);
    const thresholdLine = new Array(sessionData.length).fill(STRESS_THRESHOLDS.STRESSED);
    
    gsrChart.data.labels = labels;
    gsrChart.data.datasets[0].data = gsrValues;
    gsrChart.data.datasets[1].data = thresholdLine;
    gsrChart.update();
}

// Update pie chart
function updatePieChart() {
    const counts = countStressLevels(sessionData);
    const total = counts.relaxed + counts.normal + counts.stressed;
    
    if (total === 0) return;
    
    pieChart.data.datasets[0].data = [counts.relaxed, counts.normal, counts.stressed];
    pieChart.update();
}

// Simple linear regression prediction
function olsPredict(xArray, yArray, xFuture) {
    const n = xArray.length;
    if (n < 2) return null;
    
    const sumX = xArray.reduce((sum, x) => sum + x, 0);
    const sumY = yArray.reduce((sum, y) => sum + y, 0);
    const sumXY = xArray.reduce((sum, x, i) => sum + x * yArray[i], 0);
    const sumXX = xArray.reduce((sum, x) => sum + x * x, 0);
    
    const denominator = n * sumXX - sumX * sumX;
    if (Math.abs(denominator) < 1e-10) return null; // Avoid division by zero
    
    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;
    
    return slope * xFuture + intercept;
}

// Handle prediction
function handlePrediction() {
    if (sessionData.length < 2) {
        alert('Need at least 2 data points for prediction');
        return;
    }
    
    const minutesAheadInput = document.getElementById('minutesAhead');
    const minutesAhead = parseInt(minutesAheadInput.value);
    
    if (isNaN(minutesAhead) || minutesAhead <= 0) {
        alert('Please enter a valid positive number of minutes');
        return;
    }
    
    // Use all points for better prediction accuracy
    const xValues = sessionData.map((_, index) => index);
    const yValues = sessionData.map(d => d.gsr);
    const futureX = sessionData.length + minutesAhead;
    
    const predictedGsr = olsPredict(xValues, yValues, futureX);
    
    if (predictedGsr === null) {
        alert('Unable to make prediction - data may be constant');
        return;
    }
    
    // Ensure predicted value is not negative and reasonable
    const finalPrediction = Math.max(0, Math.min(100, predictedGsr));
    const stressClassification = classifyStress(finalPrediction);
    
    // Update UI
    document.getElementById('predictedGsr').textContent = `${finalPrediction.toFixed(2)} µS`;
    
    const stressElement = document.getElementById('predictedStress');
    stressElement.textContent = stressClassification.level;
    stressElement.className = `status status--${stressClassification.class}`;
}

// Theme toggle functionality
function toggleTheme() {
    const button = document.getElementById('themeToggle');
    
    if (currentTheme === 'light') {
        // Switch to dark mode
        document.documentElement.setAttribute('data-color-scheme', 'dark');
        button.innerHTML = '☀️ Light Mode';
        currentTheme = 'dark';
    } else {
        // Switch to light mode  
        document.documentElement.setAttribute('data-color-scheme', 'light');
        button.innerHTML = '🌙 Dark Mode';
        currentTheme = 'light';
    }
}