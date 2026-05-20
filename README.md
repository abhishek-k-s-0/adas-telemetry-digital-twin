# ADAS Vehicle Telemetry Digital Twin & Predictive Anomaly Engine 🚗📊

A localized automotive analytics framework designed to ingest high-frequency vehicle CAN bus metrics, simulate advanced mechanical and environmental physics models, and detect hazardous or erratic driving behavior using unsupervised Machine Learning.

The core pipeline functions as an edge-computing digital twin. It processes raw telemetry (velocity, steering angles, braking pressure) and dynamically calculates real-time mechanical stress profiles—such as continuous steering entropy, centrifugal lateral forces, and a multi-stage convective brake thermal dissipation model—before scoring trip integrity.

---

## 🛠️ System Engineering & Core Pipeline

The system ingests raw time-series metrics and passes them through five isolated physical and mathematical compute layers:

1. **CAN Bus Ingestion:** Captures high-frequency vehicle velocity ($km/h$), raw braking pressure ratios, steering wheel orientation degrees, and instantaneous lane tracking deviations.
2. **Environmental Twin Layer:** Implements a localized rolling window standard deviation calculation to track **Steering Entropy** while superimposing variable road-friction profiles (simulating dry surfaces vs. wet/hydroplaning zones).
3. **Mechanical Thermodynamics Core:**
   * Calculates continuous centrifugal lateral forces ($G$-force) relative to track angles and chassis velocity.
   * Runs an iterative **Brake Thermal Model** calculating rapid localized heat absorption ($\Delta Q_{\text{gain}}$) against a dynamic convective atmospheric cooling curve modeled against forward vehicle velocities.
4. **Adaptive Safety Scorer:** Computes an immediate 0-100 driver safety index by cross-referencing mechanical load stress penalties with dynamic Time-To-Collision ($TTC$) safety buffers that expand automatically when road friction drops.
5. **Unsupervised ML Audit:** Passes the combined physical feature matrices through a non-linear **Isolation Forest** ensemble architecture to isolate complex, multi-variable driving anomalies that explicit threshold rules often miss.

---

## 📈 Engineering Architecture

```text
 [Raw CAN Bus Ingest] ──► (Speed, Braking, Steering, Lane Tracking)
                                    │
                                    ▼
 [Environmental Twin] ──► (Inject Road Friction, Compute Steering Entropy)
                                    │
                                    ▼
 [Thermodynamics Core] ──► (Calculate Lateral Gs & Dynamic Brake Temperature)
                                    │
                                    ▼
 [Adaptive Safety Scorer] ──► (Calculate Safety Score & Expand Elastic TTC Buffers)
                                    │
                                    ▼
 [Isolation Forest ML] ──► (Multi-Variable Anomaly Flagging)
                                    │
                                    ▼
 [Diagnostics Engine] ──► (Matplotlib Subplots & Seaborn Root-Cause Heatmaps)
