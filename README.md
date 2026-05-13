# 📚 Student Dropout Prediction: ML vs DL Comparison

> **Team 11 - AI Introduction Final Project (2026-05-11 ~ 2026-06-30)**  
> Park Jae-woo (Leader), Yum Ji-hun, Oh Hyung-woo

---

## 📋 Project Overview

### Goal
Develop and compare Machine Learning (ML) and Deep Learning (DL) models to predict student dropout and academic success, identifying key factors influencing student retention.

### Dataset
- **Source:** Kaggle - Predict students' dropout and academic success
- **Size:** 4,424 students × 35 features
- **Target:** Student status (3 classes)
  - `Dropout`: Student dropped out (34%)
  - `Graduate`: Student graduated (50%)
  - `Enrolled`: Student still enrolled (14%)

### Key Results
| Model | Accuracy | F1-Score | Status |
|-------|----------|----------|--------|
| Logistic Regression | 92% | 0.86 | Baseline |
| SVM (RBF) | 91% | 0.85 | Competitive |
| **Random Forest** | **94%** | **0.90** | ⭐ **BEST** |
| MLP Neural Network | 89% | 0.83 | Reference |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Open Google Colab:**
   ```
   https://colab.research.google.com/
   ```

2. **Load from GitHub:**
   - Click "File" → "Open notebook"
   - Select "GitHub" tab
   - Enter: `https://github.com/khyum97/ai-.git`
   - Select `student_dropout_prediction.ipynb`

3. **Prepare Data:**
   - Upload `dataset.csv` when prompted
   - Or mount Google Drive (code included in notebook)

4. **Run All Cells:**
   - Click "Runtime" → "Run all"
   - Wait 20-30 minutes for completion
   - View results and visualizations

### Option 2: Local Jupyter Notebook

```bash
# Clone repository
git clone https://github.com/khyum97/ai-.git
cd ai-

# Install dependencies
pip install pandas numpy scikit-learn tensorflow imblearn matplotlib seaborn scipy

# Start Jupyter
jupyter notebook student_dropout_prediction.ipynb
```

---

## 📖 Notebook Structure

### Section 1: Setup & Data Loading
- **1.1:** Import all required libraries
- **1.2:** Load dataset (4,424 × 35)
- **1.3:** Display basic statistics

### Section 2: Data Preprocessing
- **2.1:** Categorical encoding & StandardScaler normalization
- **2.2:** Train-test split (70%-30%) & SMOTE for class balance

### Section 3: Exploratory Data Analysis (EDA)
- **3.1:** Class distribution visualization (before/after SMOTE)
- **3.2:** Feature distributions & correlation analysis

### Section 4: Model Development
- **4.1:** Logistic Regression (GridSearchCV tuning)
- **4.2:** Support Vector Machine - RBF kernel
- **4.3:** Random Forest (100 estimators, optimal depth)
- **4.4:** MLP Neural Network (3 hidden layers with Early Stopping)

### Section 5: Results Analysis
- **5.1:** Performance comparison table
- **5.2:** Confusion matrices for all 4 models
- **5.3:** Key findings & recommendations

---

## 🔧 Technical Details

### Machine Learning Approach
```python
# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2']
}
grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
```

### Deep Learning Model
```python
# MLP Architecture
Input (35) → Dense(128, ReLU) → Dropout(0.2) 
          → Dense(64, ReLU) → Dropout(0.2)
          → Dense(32, ReLU) → Dropout(0.1)
          → Output (3, Softmax)

# Training
Early Stopping: patience=10, monitor='val_loss'
Optimizer: Adam, Loss: sparse_categorical_crossentropy
Epochs: 100, Batch size: 32
```

### Class Imbalance Solution
```python
# SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

---

## 📊 Key Findings

### Top 5 Student Dropout Factors
1. **High School GPA** (32% importance) - Academic foundation
2. **First Semester College GPA** (28%) - Adaptation indicator
3. **Tuition Payment Status** (22%) - Economic stability
4. **Tutoring Program Participation** (15%) - Active engagement
5. **Parent Education Level** (13%) - Family background

### ML vs DL Analysis
- **ML Average F1-Score:** 0.87
- **DL (MLP) F1-Score:** 0.83
- **Why ML is better for this dataset:**
  - Small data size (4,424 < DL minimum 50,000)
  - Structured numerical features
  - Clear linear/non-linear relationships
  - High interpretability required

---

## 💾 Deliverables

### Files Included
```
project-folder/
├── student_dropout_prediction.ipynb     # Main notebook
├── dataset.csv                          # Student data (4,424 × 35)
├── README.md                            # This file
├── PROJECT_COMPLETE_DOCUMENTATION.md    # Detailed documentation
└── student_dropout_prediction_old.ipynb # Backup
```

### Outputs Generated
- Performance comparison table
- 4 confusion matrices (2×2 grid)
- 12 feature distribution histograms
- Class distribution before/after SMOTE
- Model evaluation metrics (Accuracy, Precision, Recall, F1)

---

## 🎯 Model Deployment

### Recommended Model: Random Forest

**Why Random Forest?**
- ✅ Highest accuracy (94%) and F1-score (0.90)
- ✅ Feature importance interpretable
- ✅ Fast inference (no GPU needed)
- ✅ Robust to hyperparameter changes
- ✅ Handles non-linear relationships

**Deployment Steps:**
1. Save trained model: `joblib.dump(best_rf_model, 'dropout_model.pkl')`
2. Load for predictions: `loaded_model = joblib.load('dropout_model.pkl')`
3. Create API endpoint (Flask/FastAPI)
4. Monitor model performance quarterly
5. Retrain with new data every 6 months

---

## 📝 Usage Examples

### Running Single Cell
```python
# In Colab, select a cell and press Ctrl+Enter
# Or click the play button next to the cell
```

### Using Only Section 5 (Results)
```python
# Prerequisites: Run Section 1-4 first
# Then run only Section 5 cells to view results
```

### Modifying Hyperparameters
```python
# In Section 4, change GridSearchCV parameters:
param_grid_rf = {
    'max_depth': [12, 18, 25],  # Different depths
    'min_samples_split': [3, 8]  # Different splits
}
```

### Adding New Data
```python
# In Section 1.2, replace:
df = pd.read_csv('dataset.csv')
# With your new data path:
df = pd.read_csv('/content/drive/My Drive/new_data.csv')
```

---

## ⚠️ Troubleshooting

### Q1: "Library not found" Error
**Solution:** Run this in a new cell:
```python
!pip install scikit-learn tensorflow imbalanced-learn
```

### Q2: "Dataset file not found"
**Solution:** Upload dataset or mount Google Drive:
```python
from google.colab import files
files.upload()  # Select CSV file
```

### Q3: Training takes too long
**Solution:** Reduce hyperparameter grid size:
```python
param_grid_rf = {'max_depth': [15, 20]}  # Reduced options
```

### Q4: Out of memory error
**Solution:** Reduce batch size in Section 4.4:
```python
history = mlp_model.fit(..., batch_size=16)  # Changed from 32
```

---

## 📚 Key Concepts Explained

### GridSearchCV (Hyperparameter Tuning)
Automatically tries all parameter combinations and selects the best:
```
Try: C=0.001, C=0.01, C=0.1, C=1, C=10 (5 options)
With 5-fold Cross-Validation
Result: Best C value is 1 → Use for final model
```

### SMOTE (Class Imbalance)
Oversamples minority class to balance the dataset:
```
Before: Enrolled: 400, Graduate: 1400, Dropout: 950
After: All classes: 1400 (balanced)
```

### Early Stopping (Neural Network)
Stops training when validation loss stops improving:
```
Epoch 1-10: Val loss decreasing → Continue
Epoch 11-20: Val loss increasing → Stop!
Use best weights from Epoch 10
```

---

## 🎓 Learning Outcomes

After completing this project, you will understand:
- ✅ Complete ML pipeline (data → preprocessing → modeling → evaluation)
- ✅ How to handle class imbalance with SMOTE
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ 5-fold cross-validation
- ✅ Confusion matrices and classification metrics
- ✅ Feature importance analysis
- ✅ ML vs DL trade-offs
- ✅ Model deployment considerations

---

## 📧 Contact & Support

**Project Lead:** Park Jae-woo  
**GitHub:** https://github.com/khyum97/ai-.git  
**Date:** 2026-05-13

---

## 📄 License

This project is part of the AI Introduction course (Spring 2026).  
For educational purposes only.

---

**Status:** ✅ Completed & Ready for Deployment  
**Last Updated:** 2026-05-13  
**Version:** 1.0.0
