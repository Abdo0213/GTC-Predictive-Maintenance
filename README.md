# Project 8: Predictive Maintenance for Industrial Equipment 
 
## Problem Statement 
Our operations and engineering division wants to reduce **unplanned equipment downtime**.   
Unexpected machine failures increase repair costs and cause disruptions in production and logistics. 
 
As the **ML Team**, we have been tasked with building a **predictive maintenance tool** that can forecast equipment health or estimate the remaining useful life (RUL).   
This will enable the maintenance team to take proactive action before failures occur. 
 
## Our Understanding of the Project 
We need to develop a machine learning model using **historical sensor and operational time-series data**.   
 
From our perspective, the model should be able to: 
- Estimate a machine’s **health score** or **remaining useful life (RUL)**   
- Or clearly **flag an impending failure risk** 
 
Finally, the system must be deployable through a **lightweight frontend**.   
This will allow engineers or operations staff to upload sensor logs or recent machine data and immediately receive maintenance predictions. 
 
## Data 
- **Dataset**: NASA C-MAPSS (FD001 subset).   
- **Files expected**: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (inside a `CMaps` folder).   
- **Columns defined**: `engine_id`, `time_in_cycles`, `op_setting_1..3`, `sensor_1..sensor_21`. 
 
## Work completed so far 
 
### Data preparation 
-   Loaded training, test, and RUL files from the C-MAPSS FD001 dataset. 
-   Defined column names and verified file paths. 
-   Dropped **constant-value** columns (columns with a single unique value). 
-   Calculated training **RUL** as `max_cycle - time_in_cycles` and **capped RUL at 125** cycles. 
-   Set reproducibility seeds (`PYTHONHASHSEED`, `random`, `numpy`, `tensorflow`) with `SEED_VALUE = 42`. 
   
### EDA & feature building 
-   Basic dataset inspection (shapes, per-engine cycles, feature distributions). 
-   Selected feature columns by excluding `engine_id`, `time_in_cycles`, and `RUL`. 
-   Implemented `generate_sequences()` to produce sliding-window sequences for the LSTM model. 
-   Chosen `sequence_length = 50`. 
-   Scaled features using `MinMaxScaler` (fitted on training features and applied to test features). 
-   For the test set, extracted the last `sequence_length` measurements per engine; shorter sequences are zero-padded at the beginning. 
 
### Model Training 
We implemented a mix of machine learning and deep learning models to tackle the predictive maintenance problem. 
 
-   **Classification models**: Three models (Random Forest, Gradient Boosting, and SVM) were designed to provide an early warning of potential failures, each offering different strengths in handling complex data. 
-   **Deep learning model**: An LSTM network was built to predict the Remaining Useful Life (RUL) as a continuous value. The architecture included stacked LSTM layers with dropout for regularization and dense layers for regression output. 
 
#### Training configuration 
-   **For all classification models**, we applied an 80/20 train-test split and addressed class imbalance to improve performance on the rare failure class. 
-   **The LSTM** was trained with EarlyStopping to avoid overfitting, using 100 epochs, a batch size of 32, and a 20% validation split. 
 
#### Evaluation 
-   **Classification models** were assessed using Accuracy, Precision, Recall, and F1-Score, with a strong focus on the “Failure” class due to its business relevance. 
-   **The LSTM regression model** was evaluated with RMSE, R², and the NASA C-MAPSS scoring function, which is standard for this dataset. 

--- 
## Submission scope — Current status (Model Training) 
**Note:** A detailed analysis and comparison of all model performances can be found in a separate file: `Model_Comparison.md`. 
 
### Not part of this submission (work in progress) 
- Finalized model export and deployment  

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
## ▶️ How to Run
1. Setup Environment
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
2. Run Backend (FastAPI)
    ```bash
    uvicorn main:app --reload
    ```
    API will be available at: http://127.0.0.1:8000
3. Run Frontend (Streamlit)
    ```bash
    streamlit run app.py
    ```
    Web app will be available at: http://localhost:8501