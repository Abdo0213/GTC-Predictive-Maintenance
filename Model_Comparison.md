# ⚙️ Predictive Maintenance Model Comparison  

## 📝 Project Overview  
This project implements and compares multiple machine learning models for predictive maintenance using the **NASA C-MAPSS (Turbofan Engine Degradation Simulation)** dataset.  

The goal is to **predict engine failures before they occur**, enabling proactive maintenance and avoiding costly unplanned shutdowns.  

---

## 🔬 Model Comparison  

### 1️⃣ LSTM Neural Network (Deep Learning)  
**Architecture**: Multi-layer LSTM with sequence processing  

**Key Specs**:  
- Input: 50 time steps  
- Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(64) → Dense(1)  
- Features: 17 after removing constant columns  
- Preprocessing: MinMax scaling + sequence generation  

**Performance**:  
- Accuracy: 88.85%  
- RMSE: 13.87  
- R²: 0.89  
- MAE: 10.43  
- NASA Score: 280.01  

✅ **Strengths**: Temporal patterns, high R², strong predictive power  
⚠️ **Limitations**: Heavy preprocessing, computationally intensive  

---

### 2️⃣ GRU Neural Network (Deep Learning)  
**Architecture**: Multi-layer GRU with sequence processing  

**Key Specs**: Similar to LSTM but with GRU cells (128 → 64 → 32)  

**Performance**:  
- Accuracy: 88.98%  
- RMSE: 13.80  
- R²: 0.89  
- MAE: 10.33  
- NASA Score: 299.28  

✅ **Strengths**: Similar to LSTM but faster training  
⚠️ **Limitations**: Still expensive, requires tuning  

---

### 3️⃣ Random Forest Classifier  
**Approach**: Binary classification (failure ≤ 30 cycles = 1)  

**Performance**:  
- Accuracy: 96.93%  
- Precision (Failure): 0.95  
- Recall (Failure): 0.85  
- F1-Score (Failure): 0.90  

✅ **Strengths**: Best precision/recall balance, interpretable  
⚠️ **Limitations**: Only binary output (no RUL prediction)  

---

### 4️⃣ Gradient Boosting Classifier  
**Performance**:  
- Accuracy: 96.50%  
- Precision: 0.92  
- Recall: 0.85  
- F1-Score: 0.88  

✅ **Strengths**: Good balance, error-correcting  
⚠️ **Limitations**: Prone to overfitting, binary only  

---

### 5️⃣ Support Vector Machine (SVM)  
**Performance**:  
- Accuracy: 94.86%  
- Precision: 0.76  
- Recall: **0.97**  
- F1-Score: 0.86  

✅ **Strengths**: Outstanding recall, minimal missed failures  
⚠️ **Limitations**: More false alarms, slower for big data  

---

## 📊 Model Performance Summary  

| Model            | Accuracy/R² | Precision | Recall | F1-Score | Key Advantage |
|------------------|-------------|-----------|--------|----------|---------------|
| **LSTM**         | 88.85% / 0.89 | - | - | - | Temporal patterns, RUL prediction |
| **GRU**          | 88.98% / 0.89 | - | - | - | Faster training vs LSTM |
| **Random Forest** | 96.93% | 0.95 | 0.85 | 0.90 | Best balance |
| **Gradient Boosting** | 96.50% | 0.92 | 0.85 | 0.88 | Balanced performance |
| **SVM**          | 94.86% | 0.76 | **0.97** | 0.86 | Highest recall |

---

## 🔑 Insights & Trade-offs  

- **LSTM/GRU** → RUL prediction, complex but detailed  
- **Random Forest** → Best precision-recall balance, interpretable  
- **SVM** → Highest recall, but more false alarms  

---

## ✅ Recommendations  

- **Primary** → Random Forest (best overall)  
- **Secondary** → SVM (safety-critical cases)  
- **Specialized** → LSTM/GRU (detailed RUL planning)  

---

## 🛠️ Feature Engineering Impact  

- Rolling mean/std (10-cycle windows)  
- Temporal trends  
- Removal of constant features  
- Class balancing  

---

## 🏁 Conclusion  

- **Random Forest** → Best for general use  
- **SVM** → Best for safety-critical  
- **LSTM/GRU** → Best for RUL estimation  
- **Gradient Boosting** → Solid alternative  

---

## 🚀 Future Improvements  

- **Deep Learning**: Try Transformers for sequential data  
- **Deployment**: Containerize with Docker, deploy to cloud  
- **Visualization**: Add live dashboards for sensor streams  
- **MLOps**: Add automated retraining and failure alerts  
