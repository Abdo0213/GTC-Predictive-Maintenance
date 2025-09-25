# Predictive Maintenance Model Comparison

## Project Overview

This project implements and compares multiple machine learning models for predictive maintenance using the NASA C-MAPSS (Turbofan Engine Degradation Simulation) dataset. The goal is to predict engine failures before they occur, enabling proactive maintenance and avoiding costly unplanned shutdowns.

## Model Comparison

### 1. LSTM Neural Network (Deep Learning)
**Architecture**: Multi-layer LSTM with sequence processing

**Key Specifications**:
- Input: Sequences of 50 time steps
- Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(64) → Dense(1)
- Features: 17 sensor/setting features (after removing constant columns)
- Data preprocessing: MinMax scaling, sequence generation

**Performance**:
- **Accuracy**: 88.85%
- **RMSE**: 13.87
- **R²**: 0.89
- **MAE**: 10.43
- **NASA Score**: 280.01

**Strengths**:
- Excellent at capturing temporal dependencies
- High R² score (0.89) indicates strong predictive power
- Handles sequential data naturally
- Good generalization with early stopping

**Limitations**:
- Requires extensive data preprocessing (sequence generation)
- Computationally intensive
- Complex architecture may be overkill for simpler patterns
---
### 2. GRU Neural Network (Deep Learning)
**Architecture**: Multi-layer GRU with sequence processing

**Key Specifications**:
- Input: Sequences of 50 time steps
- Architecture: GRU(128) → Dropout(0.3) → GRU(64) → Dropout(0.3) → GRU(32) → Dense(64) → Dense(1)
- Features: 17 sensor/setting features (after removing constant columns)
- Data preprocessing: MinMax scaling, sequence generation

**Performance**:
- **Accuracy**: 88.98%
- **RMSE**: 13.80
- **R²**: 0.89
- **MAE**: 10.33
- **NASA Score**: 299.28

**Strengths**:
- Captures sequential dependencies with fewer parameters than LSTM
- Similar predictive strength to LSTM
- Faster training due to simplified architecture

**Limitations**:
- Still computationally expensive compared to tree models
- Requires careful tuning and preprocessing
---
### 3. Random Forest Classifier (Tree-based Classification)
**Approach**: Ensemble method treating the problem as binary classification

**Key Specifications**:
- Target: Binary classification (failure within 30 cycles = 1, else = 0)
- Features: 47 features (original + rolling statistics)
- Estimators: 100 trees
- Class balancing: Balanced class weights

**Performance**:
- **Overall Accuracy**: 96.93%
- **Precision (Failure)**: 0.95
- **Recall (Failure)**: 0.85
- **F1-Score (Failure)**: 0.90

**Strengths**:
- Exceptional precision (95% - minimal false alarms)
- Strong recall (85% - catches most failures)
- Excellent F1-score balance
- Feature importance interpretation
- Robust to outliers

**Limitations**:
- Binary output (no RUL estimation)
- Threshold-dependent (30-cycle cutoff)
- May miss gradual degradation patterns
---
### 4. Gradient Boosting Classifier (Advanced Tree-based Classification)
**Approach**: Sequential tree building with error correction

**Key Specifications**:
- Target: Binary classification (failure within 30 cycles)
- Features: 47 features (original + rolling statistics)
- Estimators: 100
- Learning rate: 0.1, Max depth: 3

**Performance**:
- **Overall Accuracy**: 96.50%
- **Precision (Failure)**: 0.92
- **Recall (Failure)**: 0.85
- **F1-Score (Failure)**: 0.88

**Strengths**:
- High precision (92% - low false alarms)
- Good recall (85% - effective failure detection)
- Balanced performance metrics
- Sequential learning corrects previous errors

**Limitations**:
- Slightly lower precision than Random Forest
- Binary output limitation
- More prone to overfitting than Random Forest
---
### 5. Support Vector Machine (SVM) Classifier
**Approach**: Maximum margin classification with RBF kernel

**Key Specifications**:
- Kernel: RBF (Radial Basis Function)
- Features: 47 features (original + rolling statistics)
- Class balancing: Balanced class weights
- Data preprocessing: Standard scaling

**Performance**:
- **Overall Accuracy**: 94.86%
- **Precision (Failure)**: 0.76
- **Recall (Failure)**: 0.97
- **F1-Score (Failure)**: 0.86

**Strengths**:
- **Outstanding recall (97%)** - catches almost all failures
- Excellent for high-stakes scenarios where missing failures is unacceptable
- Robust mathematical foundation
- Good performance on complex decision boundaries

**Limitations**:
- Lower precision (76%) - more false alarms
- Requires feature scaling
- Computationally intensive for large datasets
- Less interpretable than tree-based methods
---
## Model Performance Summary

| Model | Accuracy/R² | Precision | Recall | F1-Score | Key Advantage |
|-------|-------------|-----------|--------|----------|---------------|
| **LSTM** | 88.85% (R²: 0.89) | - | - | - | Temporal patterns, RUL prediction |
| **GRU** | 88.98% (R²: 0.89) | - | - | - | Similar to LSTM, faster training |
| **Random Forest** | 96.93% | 0.95 | 0.85 | 0.90 | **Best balance** of precision/recall |
| **Gradient Boosting** | 96.50% | 0.92 | 0.85 | 0.88 | Good overall performance |
| **SVM** | 94.86% | 0.76 | **0.97** | 0.86 | **Highest recall** - minimal missed failures |

## Key Insights and Trade-offs

### 1. **LSTM vs GRU vs Classification Models**
- **LSTM/GRU**: Provide actual RUL values but require more complex preprocessing
- **Classification Models**: Simpler binary decision but lose granular RUL information

### 2. **Precision vs Recall Trade-off**
- **Random Forest**: Best balance - high precision with good recall
- **SVM**: Maximum recall but at cost of precision (more false alarms)
- **Gradient Boosting**: Middle ground between the two

### 3. **Practical Considerations**

**For Cost-Sensitive Operations (where false alarms are expensive)**:
- Choose **Random Forest** for optimal precision-recall balance

**For Safety-Critical Applications (where missed failures are catastrophic)**:
- Choose **SVM** for maximum recall (97% failure detection)

**For Detailed Maintenance Planning (need specific RUL values)**:
- Choose **LSTM** or **GRU** for continuous RUL estimation

## Recommendations

### Primary Recommendation: **Random Forest Classifier**
- **Why**: Best overall balance with 95% precision and 85% recall
- **Use Case**: General predictive maintenance scenarios
- **Benefits**: High accuracy, interpretable, minimal false alarms

### Secondary Recommendation: **SVM Classifier**
- **Why**: 97% recall ensures minimal missed failures
- **Use Case**: High-stakes environments where failure consequences are severe
- **Trade-off**: Accept more false alarms for maximum safety

### Specialized Use Case: **LSTM / GRU Neural Networks**
- **Why**: Provide actual RUL values and capture temporal dependencies
- **Use Case**: When detailed maintenance scheduling is required
- **Consideration**: More complex to implement and maintain

## Feature Engineering Impact

All models benefited significantly from advanced feature engineering:
- **Rolling Statistics**: 10-cycle rolling mean and standard deviation
- **Temporal Features**: Capturing trends over time windows
- **Constant Feature Removal**: Eliminated non-informative sensors
- **Class Balancing**: Addressed imbalanced failure data

## Conclusion

The choice of model depends on specific operational requirements:
- **Random Forest** offers the best general-purpose solution
- **SVM** maximizes failure detection at the cost of more false alarms  
- **LSTM/GRU** provide detailed RUL predictions for advanced planning
- **Gradient Boosting** offers solid performance as an alternative to Random Forest

Each model has proven effective for predictive maintenance, with the key being to align model selection with operational priorities and constraints.
