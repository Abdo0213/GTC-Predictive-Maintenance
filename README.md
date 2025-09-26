# ⚙️ Project 8: Predictive Maintenance for Industrial Equipment  

## 📝 Problem Statement  
Our operations and engineering division wants to reduce **unplanned equipment downtime**.  
Unexpected machine failures increase repair costs and cause disruptions in production and logistics.  

As the **ML Team**, we have been tasked with building a **predictive maintenance tool** that can forecast equipment health or estimate the remaining useful life (RUL).  
This will enable the maintenance team to take proactive action before failures occur.  

---

## 💡 Our Understanding of the Project  
We need to develop a machine learning model using **historical sensor and operational time-series data**.  

From our perspective, the model should be able to:  
- 🔋 Estimate a machine’s **health score** or **remaining useful life (RUL)**  
- 🚨 Clearly **flag an impending failure risk**  

Finally, the system must be deployable through a **lightweight frontend**.  
This will allow engineers or operations staff to upload sensor logs or recent machine data and immediately receive maintenance predictions.  

---

## 📊 Dataset  
- **Source**: [NASA C-MAPSS Dataset (FD001 subset)](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)  
- **Files**:  
  - `train_FD001.txt`  
  - `test_FD001.txt`  
  - `RUL_FD001.txt` (inside `CMaps` folder)  

**Columns Defined**:  
`engine_id`, `time_in_cycles`, `op_setting_1..3`, `sensor_1..sensor_21`  

---

## ✅ Work Completed So Far  

### 🔧 Data Preparation  
- Loaded training, test, and RUL files from the C-MAPSS FD001 dataset.  
- Dropped **constant-value** columns.  
- Calculated training **RUL** as `max_cycle - time_in_cycles` and capped RUL at 125 cycles.  
- Set reproducibility seeds (`random`, `numpy`, `tensorflow`) with `SEED_VALUE = 42`.  

### 📈 EDA & Feature Building  
- Dataset inspection (shapes, per-engine cycles, feature distributions).  
- Excluded `engine_id`, `time_in_cycles`, and `RUL` from features.  
- Implemented `generate_sequences()` with `sequence_length = 50`.  
- Scaled features with **MinMaxScaler**.  
- For test set → extracted last 50 cycles, zero-padded shorter sequences.  

### 🤖 Model Training  
- **Classification models**: Random Forest, Gradient Boosting, and SVM.  
- **Deep learning model**: LSTM for continuous RUL prediction.  

**Training setup**:  
- Classification → 80/20 split, class imbalance handled.  
- LSTM → EarlyStopping, 100 epochs, batch size 32, validation split 20%.  

**Evaluation**:  
- Classification → Accuracy, Precision, Recall, F1-Score (focus on “Failure” class 🚨).  
- LSTM → RMSE, R², NASA C-MAPSS scoring function.  

---

## 📁 Project Structure  

```bash
GTC-PREDICTIVE-MAINTENANCE/
├── Model&Scaler/                  # Saved models & scalers
│   ├── lstm_rul_model_new.h5
│   ├── lstm_rul_model.h5
│   ├── scaler_new.pkl
│   └── scaler.pkl
├── Notebooks/                     # Jupyter notebooks for experiments
│   ├── Gradient_Boosting_Model.ipynb
│   ├── GTC_fianl_FD001.ipynb
│   ├── GTC_final.ipynb
│   ├── Random_Forest_Model.ipynb
│   └── Support_Vector_Machine_Model.ipynb
├── Test/                          # Test data & scripts
│   ├── engines_csv/
│   ├── RUL_FD001.txt
│   ├── test_FD001.txt
│   └── testSplitEngine.py
├── venv/                          # Virtual environment (ignored in Git)
├── app.py                         # Streamlit frontend
├── main.py                        # FastAPI backend
├── Model_Comparison.md            # Model performance notes
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
└── .gitignore
```

---

## ▶️ How to Run  

### 1️⃣ Setup Environment  
```bash
# Create venv
python -m venv venv  

# Activate venv
# Windows (PowerShell)
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or install manually:  
```bash
pip install fastapi uvicorn streamlit tensorflow scikit-learn joblib pandas numpy requests
pip install python-multipart matplotlib seaborn
```

### 2️⃣ Run Backend (FastAPI)  
```bash
uvicorn main:app --reload
```
API available at 👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)  

### 3️⃣ Run Frontend (Streamlit)  
```bash
streamlit run app.py
```
Web app available at 👉 [http://localhost:8501](http://localhost:8501)  

---
