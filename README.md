Hi I am Ben Mbouombouo, co-founder of a starups in Silicon Valley.
Already applying AI/ML in our embedded systems for health monitoring using available CPUs and other peripherals.
Also developing NPU chips for consumer and wearable devices and applications for AI/ML on the edges for all various AI/ML architectures:
| Architecture      | Typical Use                | Edge Platform/Framework              | Notes/Performance          |
| ----------------- | -------------------------- | ------------------------------------ | -------------------------- |
| 1D-CNN/TinyML CNN | Biosignal classification   | TF Lite Micro, Edge Impulse          | Best for PPG, ECG, HRV     |
| LSTM/GRU          | Time-series/sequence       | TF Lite Micro, PyTorch Mobile        | Fall, arrhythmia, activity |
| RFR/DT/SVM        | Fast, low-power decisions  | scikit-learn, SensiML, TF Lite Micro | Hydration, anomaly         |
| Hybrid CNN-LSTM   | Multimodal fusion          | TF Lite Micro, custom hardware       | Multi-sensor wearable      |
| Tiny Transformer  | Trend analysis, adaptation | TF Lite Micro, HW accelerator        | Growing, sleep/EEG/MSD     |
| Neuromorphic      | Always-on, ultra-low-power | Ambiq, BrainChip, hardware           | Emerging, ppmW IoT/sensors |

## Glucose prediction workflow

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train and evaluate the Random Forest model using the combined PPG + demographic CSV:
   ```
   python src/glucose_prediction.py --data path/to/ppg_glucose.csv --output-model artifacts/rf_glucose.joblib
   ```
   The CSV must expose at least one `PPG*` column (raw waveform per record), demographic fields such as `age`, `gender`, and a reference `glucose` value.
3. The script extracts waveform statistics (area under the curve, energy, sample entropy, etc.), encodes demographics, trains the regressor, prints RMSE/MAE/R², and optionally saves the fitted pipeline for deployment.
