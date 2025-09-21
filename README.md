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
- Loaded training, test, and RUL files from the C-MAPSS FD001 dataset.
- Defined column names and verified file paths.
- Dropped **constant-value** columns (columns with a single unique value).
- Calculated training **RUL** as `max_cycle - time_in_cycles` and **capped RUL at 125** cycles.
- Set reproducibility seeds (`PYTHONHASHSEED`, `random`, `numpy`, `tensorflow`) with `SEED_VALUE = 42`.
  
#### EDA & feature building
- Basic dataset inspection (shapes, per-engine cycles, feature distributions).
- Selected feature columns by excluding `engine_id`, `time_in_cycles`, and `RUL`.

#### Sequence generation & scaling
- Implemented `generate_sequences(df, sequence_length, feature_cols)` to produce sliding-window sequences per `engine_id` and targets = RUL at end of sequence.
- Chosen `sequence_length = 50`.
- Scaled features using `MinMaxScaler` (fitted on training features and applied to test features).
- For the test set, extracted the last `sequence_length` measurements per engine; shorter sequences are zero-padded at the beginning.

---

#### Model development
- Built an LSTM regression model (TensorFlow / Keras) with this architecture:
  - `LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features))`
  - `Dropout(0.3)`
  - `LSTM(64, return_sequences=True, activation='tanh')`
  - `Dropout(0.3)`
  - `LSTM(32, activation='tanh')`
  - `Dropout(0.3)`
  - `Dense(64, activation='relu')`
  - `Dense(1)` (regression output for RUL)

#### Training configuration
- EarlyStopping callback: `monitor='val_loss'`, `patience=10`, `restore_best_weights=True`.
- Training parameters used in the notebook: `epochs=100`, `batch_size=32`, `validation_split=0.2`.

#### Evaluation
- Prepared test features and aligned targets from `RUL_FD001.txt`.
- Implemented evaluation metrics and scoring:
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
  - NASA C-MAPSS scoring function (`nasa_score` implemented in the notebook)
- Evaluation code runs predictions on the prepared test sequences and prints RMSE, R², and NASA score.

## Project structure (notebook sections)
1. **Data Preparation**
   - Load dataset, define columns, drop constants, calculate and cap RUL.
2. **EDA + Feature Building**
   - Inspect data and prepare feature set; set seeds.
3. **Sequence Generation & Scaling**
   - Generate sequences (`sequence_length = 50`) and apply `MinMaxScaler`.
4. **Model Training & Validation**
   - LSTM model definition and training with EarlyStopping.
5. **Evaluation**
   - Predict on test sequences, compute RMSE, R², NASA score.
6. **Deployment via Web Interface**
   - Heading present in notebook; deployment implementation not completed.
---

## Submission scope — Current status (EDA)

**Important:** the purpose of this submission is to deliver the *EDA* stage.  
Other sections in the notebook (model training, evaluation, deployment) exist as development work but are **not** part of the deliverable and remain in progress.

### Not part of this submission (work in progress)
- Hyperparameter tuning, model checkpointing, or experiment tracking
- Finalized model export and deployment 
- Production inference pipeline, monitoring, and alert thresholds


---

