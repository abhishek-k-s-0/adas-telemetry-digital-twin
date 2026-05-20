import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA GENERATION
def generate_vehicle_data(samples=1000):
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range(start='2026-04-11', periods=samples, freq='S'),
        'vehicle_speed': np.random.normal(80, 10, samples), 
        'braking_intensity': np.random.uniform(0, 1, samples), 
        'steering_angle': np.random.normal(0, 5, samples), 
        'lane_deviation': np.random.uniform(0, 0.5, samples) 
    }
    df = pd.DataFrame(data)
    df.loc[100:105, 'vehicle_speed'] = 140 # Speed Anomaly
    df.loc[500:515, 'braking_intensity'] = 0.98 # Thermal Stress Test
    return df

# 2. VIRTUAL ENVIRONMENT
def add_virtual_environment(df):
    df['road_friction'] = 0.9
    df.loc[600:900, 'road_friction'] = 0.4 # Rain zone
    df['steering_entropy'] = df['steering_angle'].rolling(window=10).std().fillna(0)
    return df

# 3. PHYSICS & THERMAL DYNAMICS
def calculate_advanced_metrics(df):
    df['lateral_g'] = (df['vehicle_speed']**2 * np.abs(df['steering_angle'])) / 150000
    current_speed_ms = df['vehicle_speed'] / 3.6
    relative_speed = current_speed_ms - (60 / 3.6)
    df['ttc'] = np.where(relative_speed > 0, 50 / relative_speed, 10.0)
    df['ttc'] = df['ttc'].clip(upper=10) 

    # Balanced Thermal Model
    ambient_temp = 30.0
    temps = [ambient_temp] 
    for i in range(1, len(df)):
        heat_gain = df.loc[i, 'braking_intensity'] * 8.0
        cooling = (temps[-1] - ambient_temp) * 0.04 + (df.loc[i, 'vehicle_speed'] * 0.01)
        temps.append(max(ambient_temp, temps[-1] + heat_gain - cooling))
    df['brake_temp'] = temps
    return df

# 4. ADAPTIVE KPIs
def process_adas_kpis(df):
    f_penalty = (1 - df['road_friction']) * 30
    t_penalty = np.where(df['brake_temp'] > 100, (df['brake_temp'] - 100) * 0.15, 0)
    df['safety_score'] = (100 - (df['lane_deviation'] * 20) - (df['braking_intensity'] * 10) - 
                          (df['lateral_g'] * 5) - (df['steering_entropy'] * 10) - f_penalty - t_penalty).clip(0, 100)
    df['dynamic_ttc_threshold'] = np.where(df['road_friction'] < 0.5, 4.0, 2.5)
    return df

# 5. ML ANOMALY DETECTION
def detect_anomalies(df):
    model = IsolationForest(contamination=0.04, random_state=42)
    features = ['vehicle_speed', 'braking_intensity', 'lane_deviation', 'lateral_g', 'ttc', 'steering_entropy', 'brake_temp']
    df['is_anomaly'] = model.fit_predict(df[features])
    return df

# 6. VISUALIZATION
def run_visualizations(df):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    ax1.plot(df['timestamp'], df['vehicle_speed'], alpha=0.6)
    ax1.scatter(df[df['is_anomaly'] == -1]['timestamp'], df[df['is_anomaly'] == -1]['vehicle_speed'], color='red', label='Anomaly')
    ax1.set_title('Vehicle Dynamics & ML Anomalies')
    
    ax2.plot(df['timestamp'], df['ttc'], color='orange')
    ax2.step(df['timestamp'], df['dynamic_ttc_threshold'], color='black', linestyle='--')
    ax2.fill_between(df['timestamp'], 0, 10, where=(df['road_friction'] < 0.5), color='gray', alpha=0.2)
    ax2.set_title('Adaptive TTC Buffer (Rain Zone)')
    
    ax3.plot(df['timestamp'], df['brake_temp'], color='red')
    ax3.axhline(y=100, color='darkred', linestyle=':')
    ax3.set_title('Brake Thermal Load (°C)')
    
    ax4.plot(df['timestamp'], df['safety_score'], color='green')
    ax4.set_ylim(0, 105); ax4.set_title('Final Driver Safety Score')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['vehicle_speed', 'lateral_g', 'ttc', 'road_friction', 'brake_temp', 'safety_score']].corr(), annot=True, cmap='RdYlGn', center=0)
    plt.title('Root-Cause Analysis'); plt.show()

# --- FINAL EXECUTION & DIAGNOSTICS ---
print("--- ADAS TRIP DIAGNOSTICS ---")
df = generate_vehicle_data()
df = detect_anomalies(process_adas_kpis(calculate_advanced_metrics(add_virtual_environment(df))))

# Summary Stats
total_anomalies = len(df[df['is_anomaly'] == -1])
avg_safety = df['safety_score'].mean()
peak_temp = df['brake_temp'].max()

print(f"Total Anomalies Detected: {total_anomalies}")
print(f"Average Trip Safety Score: {avg_safety:.2f}/100")
print(f"Peak Brake Temperature: {peak_temp:.2f}°C")
print(f"Rain Zone Intervention Ratio: {len(df[(df['ttc'] < df['dynamic_ttc_threshold']) & (df['road_friction'] < 0.5)])} incidents")
print("------------------------------")

run_visualizations(df)