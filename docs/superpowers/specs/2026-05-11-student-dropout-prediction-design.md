# Student Dropout Prediction: ML vs DL Comparative Study

**Date:** 2026-05-11  
**Project:** Team 11 - NOVA50101 Final Project  
**Team:** 박재우 (Leader), 염지훈, 오형우  
**Format:** Google Colab Jupyter Notebook  
**Timeline:** 2026-05-11 ~ 2026-06-30

---

## 1. Project Overview

### Goal
Build and compare Machine Learning and Deep Learning models to predict student dropout and academic success, identifying key factors influencing student retention.

### Dataset
- **Source:** Kaggle - Predict students' dropout and academic success
- **Size:** 4,424 rows × 35 columns
- **Target Variable:** Student status (3 classes)
  - `Dropout`: Student dropped out
  - `Graduate`: Student graduated
  - `Enrolled`: Student still enrolled
- **Features:** Student demographics, academic performance, socioeconomic factors, macroeconomic indicators

### Success Criteria
- ML models: Logistic Regression, SVM, Random Forest
- DL model: Multi-Layer Perceptron (MLP)
- Comprehensive comparison using F1-score, Precision, Recall, Accuracy
- Feature importance analysis
- Error analysis and mathematical/structural differences documented

---

## 2. Implementation Approach

### 2.1 Strategy
- **Single integrated Jupyter Notebook** in Google Colab
- **Sequential development** within parallel sections:
  1. Common preprocessing pipeline for both ML and DL
  2. Separate model development (ML in parallel with DL)
  3. Unified evaluation and comparison
- **Balanced approach:** Basic models + light hyperparameter tuning (GridSearchCV)
- **Standard EDA depth:** Key visualizations (distributions, correlations, class balance)

### 2.2 Development Phases

**Phase 1: Data Preprocessing** (5/11 ~ 5/17)
- Load and explore raw data
- Handle missing values (removal or imputation strategy)
- Categorical encoding (One-Hot or Label Encoding)
- Numerical scaling (StandardScaler)
- Address class imbalance (SMOTE or class weights)
- Apply PCA for dimensionality reduction (optional variants)
- Train/Test split (80/20)

**Phase 2: EDA & Visualization** (5/18 ~ 5/25)
- Class distribution analysis
- Feature distributions (histograms, boxplots)
- Class-wise comparisons (violin plots)
- Correlation heatmap
- Outlier detection
- Key insights summary

**Phase 3: Model Development** (5/26 ~ 6/09)
- **ML Models:**
  - Logistic Regression (baseline)
  - Support Vector Machine (non-linear)
  - Random Forest (ensemble)
- **DL Model:**
  - Multi-Layer Perceptron (2-3 hidden layers, 64-128 neurons)
- Each model: baseline → GridSearchCV tuning → cross-validation (5-fold)

**Phase 4: Evaluation & Analysis** (6/10 ~ 6/30)
- Performance metrics comparison table
- Confusion matrices and ROC curves
- Feature importance visualization
- ML vs DL trade-off analysis
- Error analysis and insights
- Final report compilation

---

## 3. Data Pipeline Architecture

```
Raw CSV Data
    ↓
[Missing Value Handling]
    ↓
[Categorical Encoding]
    ↓
[Numerical Scaling (StandardScaler)]
    ↓
[Class Imbalance Treatment]
    ├─ SMOTE (if needed)
    └─ Class Weights (backup)
    ↓
[Dimensionality Reduction (PCA)]
    ├─ Original Features (for ML models)
    └─ PCA-reduced (optional comparison)
    ↓
[Train/Test Split: 80/20]
    ↓
├─→ ML Models (processed features)
└─→ DL Models (normalized tensor data)
```

---

## 4. Model Specifications

### 4.1 Machine Learning Models

#### **Model 1: Logistic Regression**
- **Purpose:** Linear baseline, interpretable coefficients
- **Hyperparameters to tune:** Regularization strength (C), penalty type (L1/L2)
- **Cross-validation:** 5-fold
- **Output:** Probability, Feature coefficients for importance

#### **Model 2: Support Vector Machine (SVM)**
- **Purpose:** Non-linear boundary learning
- **Kernel:** RBF (radial basis function)
- **Hyperparameters to tune:** C, Gamma
- **Cross-validation:** 5-fold
- **Output:** Decision boundaries, Support vectors

#### **Model 3: Random Forest**
- **Purpose:** Ensemble model, feature importance
- **Parameters:** 100 estimators
- **Hyperparameters to tune:** Max depth, Min samples split
- **Cross-validation:** 5-fold
- **Output:** Feature importance (Gini/Entropy)

### 4.2 Deep Learning Model

#### **Multi-Layer Perceptron (MLP)**
**Architecture:**
```
Input Layer (35 features)
    ↓
Hidden Layer 1 (128 neurons, ReLU activation, Dropout 0.2)
    ↓
Hidden Layer 2 (64 neurons, ReLU activation, Dropout 0.2)
    ↓
Hidden Layer 3 (32 neurons, ReLU activation, Dropout 0.1)
    ↓
Output Layer (3 neurons, Softmax activation - 3 classes)
```

**Training:**
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 100 (with Early Stopping)
- **Validation Split:** 20%

**Regularization:**
- Dropout to prevent overfitting
- Early Stopping callback (patience=10)
- Optional batch normalization

---

## 5. Evaluation Metrics

### Per-Model Metrics
- **Accuracy:** Overall correctness
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean (primary metric for imbalanced data)
- **Confusion Matrix:** Class-wise error patterns
- **ROC-AUC (optional):** For binary comparisons

### Comparative Analysis
1. **Performance Table:** Side-by-side metrics (Accuracy, Precision, Recall, F1)
2. **Confusion Matrices:** Visual comparison of misclassifications
3. **Feature Importance:** Top 10 features from ML models
4. **Training Curves:** Loss/accuracy evolution for DL model
5. **Speed/Complexity:** Training time, model interpretability

### Error Analysis
- Misclassification patterns
- Class-specific performance gaps
- Feature relationships in errors
- Mathematical/structural differences (ML: linear/rule-based vs DL: learned representations)

---

## 6. Notebook Structure (5 Sections)

### **Section 1: Setup & Data Loading**
- Library imports (pandas, numpy, scikit-learn, TensorFlow, matplotlib, seaborn)
- Mount Google Drive (if needed for data persistence)
- Load CSV data
- Display basic info (shape, columns, data types, missing values)

### **Section 2: Data Preprocessing Pipeline**
- Missing value analysis and handling
- Categorical variable encoding
- Numerical scaling
- Class imbalance treatment (SMOTE implementation)
- PCA application (optional variants)
- Train/Test split
- Output: preprocessed datasets for ML and DL

### **Section 3: EDA & Visualization**
- Class distribution (bar chart)
- Feature distributions (20-30 key features)
- Class-wise feature comparisons
- Correlation heatmap (top correlations)
- Outlier analysis
- Summary insights

### **Section 4: ML & DL Model Development**
- **ML subsection:** Logistic Regression, SVM, Random Forest
  - Baseline training
  - GridSearchCV tuning
  - 5-fold cross-validation
  - Performance metrics
  
- **DL subsection:** MLP training
  - Model architecture definition
  - Training with callbacks
  - Learning curves
  - Performance metrics

### **Section 5: Results Comparison & Analysis**
- Unified performance comparison table
- Confusion matrices (subplots for all models)
- Feature importance visualization
- ML vs DL analysis
- Error patterns and insights
- Limitations and future work
- Final conclusions

---

## 7. Key Decisions & Trade-offs

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| **Code Format** | Google Colab Jupyter | Native environment, easy sharing, GPU support |
| **Notebook Structure** | Single integrated | Data sharing, clear pipeline flow, easier final comparison |
| **Model Depth** | Balanced (basic + light tuning) | Time efficiency while maintaining rigor |
| **EDA Level** | Standard (key visualizations) | Sufficient for insights without excessive detail |
| **Hyperparameter Tuning** | GridSearchCV (2-3 params each) | Reasonable optimization without overtraining |
| **DL Framework** | TensorFlow/Keras | Beginner-friendly, standard for educational projects |
| **Dimensionality** | Original + optional PCA | Compare performance across feature spaces |
| **Class Imbalance** | SMOTE preferred, class weights backup | Addresses real-world imbalance directly |

---

## 8. Deliverables

1. **Google Colab Notebook**
   - Executable end-to-end pipeline
   - All visualizations and results embedded
   - Documented code with explanations

2. **Analysis Outputs**
   - Performance comparison table (screenshot-exportable)
   - Feature importance rankings
   - Error analysis summary

3. **Insights Document**
   - Key findings on student dropout factors
   - ML vs DL comparison conclusions
   - Model recommendations and limitations

---

## 9. Timeline Alignment

- **5/11 ~ 5/17:** Notebook Section 2 (Preprocessing) ← **Current Phase**
- **5/18 ~ 5/25:** Notebook Section 3 (EDA)
- **5/26 ~ 6/09:** Notebook Section 4 (Model Development)
- **6/10 ~ 6/30:** Notebook Section 5 (Analysis & Report)

---

## 10. Success Criteria Checklist

- [ ] All 4 models trained and evaluated
- [ ] 5-fold cross-validation for ML models
- [ ] Early Stopping implemented for DL model
- [ ] Performance metrics calculated (Accuracy, Precision, Recall, F1)
- [ ] Feature importance extracted and visualized
- [ ] Confusion matrices generated
- [ ] ML vs DL comparison completed
- [ ] Error analysis documented
- [ ] Colab notebook fully functional and reproducible
- [ ] Final report with insights ready

---

**Status:** Design Approved - Ready for Implementation Planning
