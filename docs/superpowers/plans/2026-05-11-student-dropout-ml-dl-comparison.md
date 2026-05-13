# Student Dropout Prediction: ML vs DL Comparative Study - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end Google Colab notebook that loads student data, preprocesses it, performs EDA, trains ML and DL models, and compares their performance for predicting student dropout.

**Architecture:** Single Jupyter notebook with 5 sequential sections: (1) Setup & Data Loading, (2) Data Preprocessing Pipeline, (3) EDA & Visualization, (4) ML & DL Model Development, (5) Results Comparison & Analysis. All processing happens in-memory within the notebook.

**Tech Stack:** 
- Google Colab (free GPU/TPU)
- pandas, numpy (data manipulation)
- scikit-learn (ML models: LogReg, SVM, Random Forest)
- TensorFlow/Keras (DL: MLP)
- matplotlib, seaborn (visualization)

---

## File Structure

```
Google Colab Notebook: student_dropout_prediction.ipynb
├─ Section 1: Setup & Data Loading
│  ├─ Library imports
│  ├─ Mount Google Drive (optional)
│  └─ Load CSV from colab_data/dataset.csv
├─ Section 2: Data Preprocessing
│  ├─ Missing value handling
│  ├─ Categorical encoding
│  ├─ Scaling
│  ├─ SMOTE for class imbalance
│  ├─ PCA (optional)
│  └─ Train/Test split
├─ Section 3: EDA & Visualization
│  ├─ Class distribution
│  ├─ Feature distributions
│  ├─ Correlation analysis
│  └─ Outlier detection
├─ Section 4: ML & DL Models
│  ├─ Logistic Regression
│  ├─ SVM
│  ├─ Random Forest
│  └─ Multi-Layer Perceptron
└─ Section 5: Results & Analysis
   ├─ Performance comparison table
   ├─ Confusion matrices
   ├─ Feature importance
   └─ Final insights
```

---

## Task Breakdown

### Task 1: Create Colab Notebook & Section 1 (Setup & Data Loading)

**Deliverable:** Google Colab notebook with working data loading

- [ ] **Step 1: Create Google Colab notebook**

Go to https://colab.research.google.com and create a new notebook named `student_dropout_prediction.ipynb`

- [ ] **Step 2: Add title cell**

Cell Type: Markdown
```markdown
# Student Dropout Prediction: ML vs DL Comparative Study
**Team 11:** 박재우, 염지훈, 오형우  
**Course:** NOVA50101 - Introduction to AI  
**Date:** 2026-05-11
```

- [ ] **Step 3: Create Section 1 header**

Cell Type: Markdown
```markdown
## Section 1: Setup & Data Loading

Import required libraries and load dataset.
```

- [ ] **Step 4: Write library imports cell**

Cell Type: Code
```python
# Data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# DL Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Warnings
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully")
```

- [ ] **Step 5: Create data loading cell**

Cell Type: Code
```python
# Upload dataset
from google.colab import files
print("Upload dataset.csv:")
uploaded = files.upload()

# Verify upload
file_name = list(uploaded.keys())[0]
print(f"✓ Uploaded: {file_name}")

# Load data
df = pd.read_csv(file_name)

# Display basic info
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📋 Column Names & Types:\n{df.dtypes}")
print(f"\n❓ Missing Values:\n{df.isnull().sum()}")
print(f"\n🎯 Target Distribution:\n{df['Target'].value_counts()}")
print(f"\n📄 First 5 rows:\n{df.head()}")
```

- [ ] **Step 6: Run data loading cell and verify output**

Expected Output:
```
✓ Uploaded: dataset.csv
📊 Dataset Shape: (4424, 35)
📋 Column Names & Types: (35 columns listed with types)
❓ Missing Values: (summary of missing values)
🎯 Target Distribution: 
Dropout      1421
Graduate     2209
Enrolled      794
🎯 First 5 rows: (sample data)
```

- [ ] **Step 7: Commit progress (save notebook)**

In Colab: File → Save (Ctrl+S)
File path will be: `Google Drive/Colab Notebooks/student_dropout_prediction.ipynb`

---

### Task 2: Section 2 - Data Preprocessing Pipeline (Part A: Exploration & Preparation)

**Deliverable:** Missing value analysis and encoding strategy defined

- [ ] **Step 1: Create Section 2 header**

Cell Type: Markdown
```markdown
## Section 2: Data Preprocessing Pipeline

Handle missing values, encode categorical variables, scale numerical features, address class imbalance, and prepare train/test sets.
```

- [ ] **Step 2: Add missing value analysis cell**

Cell Type: Code
```python
# Detailed missing value analysis
print("=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_summary) > 0:
    print(missing_summary.to_string(index=False))
else:
    print("✓ No missing values found!")

# Display data types
print("\n" + "=" * 60)
print("DATA TYPE SUMMARY")
print("=" * 60)
print(df.dtypes.value_counts())
print(f"\nNumerical Columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical Columns: {df.select_dtypes(include=['object']).columns.tolist()}")
```

- [ ] **Step 3: Run exploration cell**

Expected Output:
```
MISSING VALUE ANALYSIS
(summary of any missing values)
✓ No missing values found! (or: missing value counts)

DATA TYPE SUMMARY
int64    30
object    5
Numerical Columns: [list of numeric cols]
Categorical Columns: [list of categorical cols]
```

- [ ] **Step 4: Add data cleaning cell**

Cell Type: Code
```python
# Create working copy
df_processed = df.copy()

# Remove any complete duplicates (if any)
initial_rows = len(df_processed)
df_processed = df_processed.drop_duplicates()
print(f"✓ Removed {initial_rows - len(df_processed)} duplicate rows")

# Verify Target variable
print(f"\n✓ Target Classes: {df_processed['Target'].unique()}")
print(f"✓ Class Distribution:\n{df_processed['Target'].value_counts()}")

# Display cleaned dataset info
print(f"\n✓ Cleaned Dataset Shape: {df_processed.shape}")
```

- [ ] **Step 5: Run cleaning cell and verify**

Expected Output shows cleaned data shape and class distribution.

---

### Task 3: Section 2 - Data Preprocessing Pipeline (Part B: Encoding & Scaling)

**Deliverable:** Encoded and scaled features ready for modeling

- [ ] **Step 1: Add categorical encoding cell**

Cell Type: Code
```python
# Identify categorical and numerical columns
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

# Remove target from feature columns
if 'Target' in categorical_cols:
    categorical_cols.remove('Target')
if 'Target' in numerical_cols:
    numerical_cols.remove('Target')

print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables with Label Encoding (for interpretability)
label_encoders = {}
df_encoded = df_processed.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode target variable
target_encoder = LabelEncoder()
df_encoded['Target'] = target_encoder.fit_transform(df_processed['Target'])
label_encoders['Target'] = target_encoder

print(f"\n✓ Target Encoding: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
print(f"\n✓ All categorical variables encoded")
print(f"\n✓ Encoded Dataset:\n{df_encoded.head()}")
```

- [ ] **Step 2: Run encoding cell**

Expected Output shows encoded values for categorical columns and target.

- [ ] **Step 3: Add scaling cell**

Cell Type: Code
```python
# Separate features and target
X = df_encoded.drop('Target', axis=1)
y = df_encoded['Target']

print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")
print(f"Feature Columns: {X.columns.tolist()}")

# Initialize StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"\n✓ StandardScaler fitted")
print(f"✓ Scaled Features - Mean (should be ~0): {X_scaled.mean().mean():.6f}")
print(f"✓ Scaled Features - Std (should be ~1): {X_scaled.std().mean():.6f}")
print(f"\n✓ Scaled Dataset:\n{X_scaled.head()}")
```

- [ ] **Step 4: Run scaling cell and verify**

Expected Output shows scaled data with mean ~0 and std ~1.

---

### Task 4: Section 2 - Data Preprocessing Pipeline (Part C: Train/Test Split & Class Imbalance)

**Deliverable:** Train and test sets with balanced classes ready for modeling

- [ ] **Step 1: Add train/test split cell**

Cell Type: Code
```python
# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"✓ Train/Test Split (80/20):")
print(f"  Train Set: {X_train.shape}")
print(f"  Test Set: {X_test.shape}")

print(f"\n✓ Train Set Class Distribution:")
print(y_train.value_counts().sort_index())
print(f"\n✓ Test Set Class Distribution:")
print(y_test.value_counts().sort_index())

# Store original sets for comparison
X_train_original = X_train.copy()
y_train_original = y_train.copy()
```

- [ ] **Step 2: Run split cell**

Expected Output shows train/test shapes and class distributions.

- [ ] **Step 3: Add SMOTE (class imbalance handling) cell**

Cell Type: Code
```python
# Apply SMOTE to training set only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"✓ SMOTE Applied to Training Set:")
print(f"  Original Train Shape: {X_train.shape}")
print(f"  Balanced Train Shape: {X_train_balanced.shape}")

print(f"\n✓ Original Train Class Distribution:")
print(y_train_original.value_counts().sort_index())

print(f"\n✓ Balanced Train Class Distribution:")
print(y_train_balanced.value_counts().sort_index())

# Convert back to DataFrame for consistency
X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
```

- [ ] **Step 4: Run SMOTE cell and verify**

Expected Output shows balanced training set with equal class representation.

- [ ] **Step 5: Add summary cell**

Cell Type: Code
```python
print("=" * 60)
print("PREPROCESSING PIPELINE SUMMARY")
print("=" * 60)
print(f"✓ Original Dataset: {df.shape}")
print(f"✓ Features Shape: {X_scaled.shape}")
print(f"✓ Target Classes: {target_encoder.classes_.tolist()}")
print(f"✓ Train Set (balanced): {X_train_balanced.shape}")
print(f"✓ Test Set: {X_test.shape}")
print(f"\n✓ Ready for modeling!")
```

- [ ] **Step 6: Save notebook progress**

File → Save (Ctrl+S)

---

### Task 5: Section 3 - EDA & Visualization (Part A: Class & Feature Distributions)

**Deliverable:** Comprehensive EDA visualizations

- [ ] **Step 1: Create Section 3 header**

Cell Type: Markdown
```markdown
## Section 3: Exploratory Data Analysis (EDA) & Visualization

Analyze class distributions, feature patterns, correlations, and outliers.
```

- [ ] **Step 2: Add class distribution visualization cell**

Cell Type: Code
```python
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target Class Distribution (Original)
target_counts = df_processed['Target'].value_counts()
ax = axes[0, 0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax.bar(target_counts.index, target_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Student Status', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Original Class Distribution', fontsize=13, fontweight='bold')
for i, v in enumerate(target_counts.values):
    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')

# 2. Target Class Distribution (Percentage)
ax = axes[0, 1]
percentages = (target_counts / len(df_processed) * 100).round(2)
ax.pie(percentages.values, labels=percentages.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Class Distribution (%)', fontsize=13, fontweight='bold')

# 3. Train Class Distribution (Original)
train_counts = y_train_original.value_counts().sort_index()
ax = axes[1, 0]
ax.bar(target_encoder.classes_[train_counts.index], train_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Student Status', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Original Train Set Distribution', fontsize=13, fontweight='bold')
for i, v in enumerate(train_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontweight='bold')

# 4. Train Class Distribution (After SMOTE)
train_counts_balanced = y_train_balanced.value_counts().sort_index()
ax = axes[1, 1]
ax.bar(target_encoder.classes_[train_counts_balanced.index], train_counts_balanced.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Student Status', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Balanced Train Set Distribution (After SMOTE)', fontsize=13, fontweight='bold')
for i, v in enumerate(train_counts_balanced.values):
    ax.text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("✓ Class distribution visualizations generated")
```

- [ ] **Step 3: Run visualization cell**

Expected Output shows 4 subplots with class distributions.

- [ ] **Step 4: Add feature distributions cell**

Cell Type: Code
```python
# Select top 12 numerical features for visualization
# (using original data for interpretation)
feature_cols = X.columns.tolist()[:12]  # First 12 features

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    
    # Create histogram with KDE
    ax.hist(X[col], bins=30, alpha=0.6, color='steelblue', edgecolor='black')
    ax.set_xlabel(col, fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"✓ Distribution plots for {len(feature_cols)} features generated")
```

- [ ] **Step 5: Run feature distribution cell**

Expected Output shows 12 histograms of numerical features.

---

### Task 6: Section 3 - EDA & Visualization (Part B: Correlation & Outliers)

**Deliverable:** Correlation analysis and outlier detection

- [ ] **Step 1: Add correlation heatmap cell**

Cell Type: Code
```python
# Calculate correlation matrix
correlation_matrix = X.corr()

# Select top correlated features with target (using original encoding)
# Merge features and target for correlation calculation
X_y = X.copy()
X_y['Target'] = y

target_correlation = X_y.corr()['Target'].drop('Target').sort_values(ascending=False)

print("✓ Top 10 Features Most Correlated with Dropout:")
print(target_correlation.head(10))

print("\n✓ Top 10 Features Least Correlated (Most Negative) with Dropout:")
print(target_correlation.tail(10))

# Plot heatmap of top correlations
fig, ax = plt.subplots(figsize=(14, 10))

# Select top 15 features by correlation magnitude
top_features = list(target_correlation.abs().nlargest(15).index)
corr_subset = correlation_matrix.loc[top_features, top_features]

sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Top 15 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Correlation heatmap generated")
```

- [ ] **Step 2: Run correlation cell**

Expected Output shows top correlated features and heatmap.

- [ ] **Step 3: Add outlier detection cell**

Cell Type: Code
```python
# Detect outliers using IQR method
def detect_outliers_iqr(data):
    outlier_indices = []
    
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
        outlier_indices.extend(col_outliers)
    
    return list(set(outlier_indices))

outlier_indices = detect_outliers_iqr(X_train)
print(f"✓ Outliers Detected (IQR method): {len(outlier_indices)} rows")
print(f"  Percentage of Training Set: {len(outlier_indices)/len(X_train)*100:.2f}%")

# Boxplot for selected features
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

selected_features = X.columns.tolist()[:6]
for idx, col in enumerate(selected_features):
    ax = axes[idx]
    ax.boxplot(X[col], vert=True)
    ax.set_ylabel(col, fontsize=10, fontweight='bold')
    ax.set_title(f'Boxplot: {col}', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Outlier analysis complete")
```

- [ ] **Step 4: Run outlier detection cell**

Expected Output shows outlier count and boxplots.

- [ ] **Step 5: Add EDA summary cell**

Cell Type: Code
```markdown
print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"✓ Dataset: {len(df_processed)} students, {len(X.columns)} features")
print(f"✓ Classes: Dropout ({(df_processed['Target']=='Dropout').sum()}), "
      f"Graduate ({(df_processed['Target']=='Graduate').sum()}), "
      f"Enrolled ({(df_processed['Target']=='Enrolled').sum()})")
print(f"✓ Most Correlated Feature: {target_correlation.index[0]} ({target_correlation.iloc[0]:.3f})")
print(f"✓ Outliers: {len(outlier_indices)} ({len(outlier_indices)/len(X_train)*100:.2f}%)")
print(f"✓ EDA Complete - Ready for Modeling!")
```

- [ ] **Step 6: Save notebook**

File → Save (Ctrl+S)

---

### Task 7: Section 4 - ML & DL Models (Part A: Logistic Regression & SVM)

**Deliverable:** Trained and evaluated Logistic Regression and SVM models

- [ ] **Step 1: Create Section 4 header**

Cell Type: Markdown
```markdown
## Section 4: Machine Learning & Deep Learning Models

Train and evaluate 4 models: Logistic Regression, SVM, Random Forest, and Neural Network.
```

- [ ] **Step 2: Add Logistic Regression cell**

Cell Type: Code
```python
from sklearn.metrics import confusion_matrix

print("=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

# Initialize model
lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')

# GridSearchCV for hyperparameter tuning
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2']
}

grid_lr = GridSearchCV(lr_model, param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_lr.fit(X_train_balanced, y_train_balanced)

print(f"✓ Best Parameters: {grid_lr.best_params_}")
print(f"✓ Best CV Score (F1-weighted): {grid_lr.best_score_:.4f}")

# Predict on test set
y_pred_lr = grid_lr.predict(X_test)

# Calculate metrics
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print(f"\n✓ Test Set Performance:")
print(f"  Accuracy:  {acc_lr:.4f}")
print(f"  Precision: {prec_lr:.4f}")
print(f"  Recall:    {rec_lr:.4f}")
print(f"  F1-Score:  {f1_lr:.4f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"\n✓ Confusion Matrix:\n{cm_lr}")

# Store results
models_results = {
    'Logistic Regression': {
        'model': grid_lr.best_estimator_,
        'predictions': y_pred_lr,
        'accuracy': acc_lr,
        'precision': prec_lr,
        'recall': rec_lr,
        'f1': f1_lr,
        'confusion_matrix': cm_lr
    }
}
```

- [ ] **Step 3: Run Logistic Regression cell**

Expected Output shows best parameters and metrics (Accuracy, Precision, Recall, F1).

- [ ] **Step 4: Add SVM cell**

Cell Type: Code
```python
print("\n" + "=" * 70)
print("MODEL 2: SUPPORT VECTOR MACHINE (SVM)")
print("=" * 70)

# Initialize SVM
svm_model = SVC(kernel='rbf', random_state=42)

# GridSearchCV for hyperparameter tuning
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1]
}

grid_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_svm.fit(X_train_balanced, y_train_balanced)

print(f"✓ Best Parameters: {grid_svm.best_params_}")
print(f"✓ Best CV Score (F1-weighted): {grid_svm.best_score_:.4f}")

# Predict on test set
y_pred_svm = grid_svm.predict(X_test)

# Calculate metrics
acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
rec_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print(f"\n✓ Test Set Performance:")
print(f"  Accuracy:  {acc_svm:.4f}")
print(f"  Precision: {prec_svm:.4f}")
print(f"  Recall:    {rec_svm:.4f}")
print(f"  F1-Score:  {f1_svm:.4f}")

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(f"\n✓ Confusion Matrix:\n{cm_svm}")

# Store results
models_results['SVM'] = {
    'model': grid_svm.best_estimator_,
    'predictions': y_pred_svm,
    'accuracy': acc_svm,
    'precision': prec_svm,
    'recall': rec_svm,
    'f1': f1_svm,
    'confusion_matrix': cm_svm
}
```

- [ ] **Step 5: Run SVM cell**

Expected Output shows best parameters and metrics.

- [ ] **Step 6: Save notebook**

File → Save (Ctrl+S)

---

### Task 8: Section 4 - ML & DL Models (Part B: Random Forest)

**Deliverable:** Trained Random Forest model with feature importance

- [ ] **Step 1: Add Random Forest cell**

Cell Type: Code
```python
print("\n" + "=" * 70)
print("MODEL 3: RANDOM FOREST")
print("=" * 70)

# Initialize Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# GridSearchCV for hyperparameter tuning
param_grid_rf = {
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
}

grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train_balanced, y_train_balanced)

print(f"✓ Best Parameters: {grid_rf.best_params_}")
print(f"✓ Best CV Score (F1-weighted): {grid_rf.best_score_:.4f}")

# Predict on test set
y_pred_rf = grid_rf.predict(X_test)

# Calculate metrics
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\n✓ Test Set Performance:")
print(f"  Accuracy:  {acc_rf:.4f}")
print(f"  Precision: {prec_rf:.4f}")
print(f"  Recall:    {rec_rf:.4f}")
print(f"  F1-Score:  {f1_rf:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\n✓ Confusion Matrix:\n{cm_rf}")

# Feature Importance
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': grid_rf.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n✓ Top 10 Important Features:")
print(feature_importance_rf.head(10).to_string(index=False))

# Store results
models_results['Random Forest'] = {
    'model': grid_rf.best_estimator_,
    'predictions': y_pred_rf,
    'accuracy': acc_rf,
    'precision': prec_rf,
    'recall': rec_rf,
    'f1': f1_rf,
    'confusion_matrix': cm_rf,
    'feature_importance': feature_importance_rf
}
```

- [ ] **Step 2: Run Random Forest cell**

Expected Output shows metrics and top 10 features.

- [ ] **Step 3: Add feature importance visualization cell**

Cell Type: Code
```python
# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))

top_features = feature_importance_rf.head(15)
ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Features - Random Forest', fontsize=13, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

print("✓ Feature importance visualization generated")
```

- [ ] **Step 4: Run visualization cell**

Expected Output shows horizontal bar chart of feature importance.

- [ ] **Step 5: Save notebook**

File → Save (Ctrl+S)

---

### Task 9: Section 4 - ML & DL Models (Part C: Neural Network - MLP)

**Deliverable:** Trained and evaluated multi-layer perceptron neural network

- [ ] **Step 1: Add Neural Network definition cell**

Cell Type: Code
```python
print("\n" + "=" * 70)
print("MODEL 4: MULTI-LAYER PERCEPTRON (MLP) - DEEP LEARNING")
print("=" * 70)

# Normalize features for DL (0-1 range)
from sklearn.preprocessing import MinMaxScaler

scaler_dl = MinMaxScaler()
X_train_dl = scaler_dl.fit_transform(X_train_balanced)
X_test_dl = scaler_dl.transform(X_test)

# Convert to tensors
X_train_dl = tf.convert_to_tensor(X_train_dl, dtype=tf.float32)
X_test_dl = tf.convert_to_tensor(X_test_dl, dtype=tf.float32)
y_train_dl = tf.convert_to_tensor(y_train_balanced, dtype=tf.int32)
y_test_dl = tf.convert_to_tensor(y_test, dtype=tf.int32)

print(f"✓ Data prepared for Deep Learning")
print(f"  Train: {X_train_dl.shape}")
print(f"  Test: {X_test_dl.shape}")

# Define MLP architecture
mlp_model = models.Sequential([
    layers.Input(shape=(X_train_dl.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

print(f"\n✓ MLP Architecture:")
mlp_model.summary()

# Compile model
mlp_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

- [ ] **Step 2: Run model definition cell**

Expected Output shows model summary with layer details.

- [ ] **Step 3: Add training cell**

Cell Type: Code
```python
# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
print("\n✓ Training MLP (this may take a few minutes)...")
history = mlp_model.fit(
    X_train_dl, y_train_dl,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0  # Suppress output for cleaner notebook
)

print(f"✓ Training complete in {len(history.history['loss'])} epochs")
print(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

- [ ] **Step 4: Run training cell**

Expected Output shows training progress and final metrics.

- [ ] **Step 5: Add learning curves visualization cell**

Cell Type: Code
```python
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Loss curves
ax = axes[0]
ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax.set_title('MLP Training & Validation Loss', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Accuracy curves
ax = axes[1]
ax.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_title('MLP Training & Validation Accuracy', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Learning curves visualization generated")
```

- [ ] **Step 6: Run learning curves cell**

Expected Output shows 2 plots of loss and accuracy curves.

- [ ] **Step 7: Add MLP evaluation cell**

Cell Type: Code
```python
# Predict on test set
y_pred_mlp_probs = mlp_model.predict(X_test_dl)
y_pred_mlp = np.argmax(y_pred_mlp_probs, axis=1)

# Calculate metrics
acc_mlp = accuracy_score(y_test, y_pred_mlp)
prec_mlp = precision_score(y_test, y_pred_mlp, average='weighted', zero_division=0)
rec_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

print(f"\n✓ Test Set Performance:")
print(f"  Accuracy:  {acc_mlp:.4f}")
print(f"  Precision: {prec_mlp:.4f}")
print(f"  Recall:    {rec_mlp:.4f}")
print(f"  F1-Score:  {f1_mlp:.4f}")

# Confusion Matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print(f"\n✓ Confusion Matrix:\n{cm_mlp}")

# Store results
models_results['MLP'] = {
    'model': mlp_model,
    'predictions': y_pred_mlp,
    'accuracy': acc_mlp,
    'precision': prec_mlp,
    'recall': rec_mlp,
    'f1': f1_mlp,
    'confusion_matrix': cm_mlp
}

print("\n✓ All 4 models trained successfully!")
```

- [ ] **Step 8: Run evaluation cell**

Expected Output shows MLP metrics.

- [ ] **Step 9: Save notebook**

File → Save (Ctrl+S)

---

### Task 10: Section 5 - Results Comparison & Analysis (Part A: Performance Metrics)

**Deliverable:** Comprehensive performance comparison table and confusion matrices

- [ ] **Step 1: Create Section 5 header**

Cell Type: Markdown
```markdown
## Section 5: Results Comparison & Analysis

Compare model performance, analyze errors, extract insights, and generate final conclusions.
```

- [ ] **Step 2: Add performance comparison table cell**

Cell Type: Code
```python
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(models_results.keys()),
    'Accuracy': [models_results[m]['accuracy'] for m in models_results],
    'Precision': [models_results[m]['precision'] for m in models_results],
    'Recall': [models_results[m]['recall'] for m in models_results],
    'F1-Score': [models_results[m]['f1'] for m in models_results]
})

print("\n✓ Performance Metrics Table:")
print(comparison_df.to_string(index=False))

# Find best model for each metric
print(f"\n✓ Best Models by Metric:")
print(f"  Accuracy:  {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']} ({comparison_df['Accuracy'].max():.4f})")
print(f"  Precision: {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Model']} ({comparison_df['Precision'].max():.4f})")
print(f"  Recall:    {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']} ({comparison_df['Recall'].max():.4f})")
print(f"  F1-Score:  {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']} ({comparison_df['F1-Score'].max():.4f})")
```

- [ ] **Step 3: Run comparison table cell**

Expected Output shows performance comparison table and best models.

- [ ] **Step 4: Add confusion matrices visualization cell**

Cell Type: Code
```python
# Plot confusion matrices for all models
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

model_names = list(models_results.keys())
for idx, model_name in enumerate(model_names):
    ax = axes[idx]
    cm = models_results[model_name]['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_,
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("✓ Confusion matrices generated")
```

- [ ] **Step 5: Run confusion matrices cell**

Expected Output shows 4 confusion matrices.

- [ ] **Step 6: Add detailed error analysis cell**

Cell Type: Code
```python
# Classification Reports for each model
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 70)

for model_name in models_results:
    y_pred = models_results[model_name]['predictions']
    print(f"\n{'='*70}")
    print(f"{model_name.upper()}")
    print(f"{'='*70}")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
```

- [ ] **Step 7: Run classification reports cell**

Expected Output shows detailed precision/recall/F1 for each class and model.

- [ ] **Step 8: Save notebook**

File → Save (Ctrl+S)

---

### Task 11: Section 5 - Results Comparison & Analysis (Part B: Feature Importance & Insights)

**Deliverable:** Feature importance analysis and final insights

- [ ] **Step 1: Add feature importance comparison cell**

Cell Type: Code
```python
# Extract feature importance from Logistic Regression (coefficients)
lr_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': grid_lr.best_estimator_.coef_[0]  # Use first class coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Logistic Regression Coefficients
ax = axes[0]
top_lr = lr_coefficients.head(15)
colors_lr = ['red' if x < 0 else 'green' for x in top_lr['Coefficient']]
ax.barh(range(len(top_lr)), top_lr['Coefficient'].values, color=colors_lr, edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(top_lr)))
ax.set_yticklabels(top_lr['Feature'].values)
ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Features - Logistic Regression\n(Red=Negative, Green=Positive)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Random Forest Feature Importance (already computed)
ax = axes[1]
top_rf = feature_importance_rf.head(15)
ax.barh(range(len(top_rf)), top_rf['Importance'].values, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(top_rf)))
ax.set_yticklabels(top_rf['Feature'].values)
ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Features - Random Forest', fontsize=12, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

print("✓ Feature importance comparison generated")
```

- [ ] **Step 2: Run feature importance cell**

Expected Output shows 2 feature importance plots.

- [ ] **Step 3: Add model comparison visualization cell**

Cell Type: Code
```python
# Create radar-like comparison visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Normalize metrics to 0-1 scale for visualization
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.2

for idx, model_name in enumerate(comparison_df['Model']):
    values = comparison_df[comparison_df['Model'] == model_name][metrics].values[0]
    ax.bar(x + idx*width, values, width, label=model_name, edgecolor='black', alpha=0.8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - All Metrics', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Model comparison visualization generated")
```

- [ ] **Step 4: Run model comparison visualization cell**

Expected Output shows grouped bar chart comparing all metrics.

- [ ] **Step 5: Add ML vs DL analysis cell**

Cell Type: Code
```python
print("\n" + "=" * 70)
print("MACHINE LEARNING vs DEEP LEARNING ANALYSIS")
print("=" * 70)

# Aggregate ML models
ml_models = ['Logistic Regression', 'SVM', 'Random Forest']
dl_models = ['MLP']

ml_scores = comparison_df[comparison_df['Model'].isin(ml_models)][['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
dl_scores = comparison_df[comparison_df['Model'].isin(dl_models)][['Accuracy', 'Precision', 'Recall', 'F1-Score']].iloc[0]

print(f"\n✓ Machine Learning (Average of 3 models):")
for metric in ml_scores.index:
    print(f"  {metric}: {ml_scores[metric]:.4f}")

print(f"\n✓ Deep Learning (MLP):")
for metric in dl_scores.index:
    print(f"  {metric}: {dl_scores[metric]:.4f}")

print(f"\n✓ Comparison:")
for metric in ml_scores.index:
    diff = dl_scores[metric] - ml_scores[metric]
    sign = "+" if diff > 0 else ""
    print(f"  {metric}: DL {sign}{diff:.4f} vs ML")

# Structural differences
print(f"\n✓ Key Structural Differences:")
print(f"  ML Models: Rule-based/linear boundaries, interpretable coefficients")
print(f"  DL Model: Learned non-linear transformations through hidden layers")
print(f"  ML Advantage: Faster training, easier interpretation")
print(f"  DL Advantage: Can capture complex non-linear patterns, dropout prevents overfitting")
```

- [ ] **Step 6: Run ML vs DL analysis cell**

Expected Output shows comparison of ML vs DL performance and insights.

---

### Task 12: Section 5 - Final Summary & Report

**Deliverable:** Comprehensive final report with key findings and recommendations

- [ ] **Step 1: Add key findings cell**

Cell Type: Code
```python
print("=" * 70)
print("KEY FINDINGS & INSIGHTS")
print("=" * 70)

# Best overall model
best_f1_idx = comparison_df['F1-Score'].idxmax()
best_model = comparison_df.iloc[best_f1_idx]['Model']
best_f1 = comparison_df.iloc[best_f1_idx]['F1-Score']

print(f"\n✓ Best Performing Model: {best_model}")
print(f"  F1-Score: {best_f1:.4f}")

# Top predictive features
print(f"\n✓ Most Important Factors for Dropout Prediction:")
print(f"  (Based on Random Forest Feature Importance)")
for idx, row in feature_importance_rf.head(5).iterrows():
    print(f"  - {row['Feature']}: {row['Importance']:.4f}")

# Class-wise analysis
print(f"\n✓ Class-wise Performance:")
print(f"  (Dropout is the primary focus for early intervention)")

# Dropout class performance
dropout_idx = np.where(target_encoder.classes_ == 'Dropout')[0][0]
print(f"  Dropout Recall (best model): {models_results[best_model]['confusion_matrix'][dropout_idx][dropout_idx] / models_results[best_model]['confusion_matrix'][dropout_idx].sum():.4f}")

# Model complexity vs performance
print(f"\n✓ Model Complexity Analysis:")
print(f"  Logistic Regression: Simple, fast, baseline")
print(f"  SVM: Non-linear boundaries, moderate complexity")
print(f"  Random Forest: Ensemble, interpretable features, good balance")
print(f"  MLP: Deep learning, more parameters, higher complexity")
```

- [ ] **Step 2: Run key findings cell**

Expected Output shows main discoveries and insights.

- [ ] **Step 3: Add limitations & recommendations cell**

Cell Type: Code
```python
print("\n" + "=" * 70)
print("PROJECT LIMITATIONS & RECOMMENDATIONS")
print("=" * 70)

print(f"\n✓ Limitations:")
print(f"  1. Dataset represents single institution (generalization risk)")
print(f"  2. No temporal dimension (static snapshot of students)")
print(f"  3. Macro-economic indicators not up-to-date")
print(f"  4. Limited DL architecture exploration (only basic MLP)")
print(f"  5. Balanced training set may not reflect real-world deployment")

print(f"\n✓ Future Improvements:")
print(f"  1. Collect multi-institution data for broader validation")
print(f"  2. Implement time-series models (LSTM) for temporal patterns")
print(f"  3. Try advanced DL architectures (attention mechanisms, ensemble DL)")
print(f"  4. Cross-validate with external datasets")
print(f"  5. Deploy model with confidence intervals for business decision-making")

print(f"\n✓ Recommendations for Practical Use:")
print(f"  1. Use ensemble of {best_model} with confidence thresholds")
print(f"  2. Focus on top 5 features for intervention programs")
print(f"  3. Update model quarterly with new student data")
print(f"  4. Monitor model drift through regular validation")
print(f"  5. Combine predictions with academic advisor judgment")
```

- [ ] **Step 4: Run limitations cell**

Expected Output shows project limitations and recommendations.

- [ ] **Step 5: Add final summary cell**

Cell Type: Code
```python
print("\n" + "=" * 70)
print("PROJECT COMPLETION SUMMARY")
print("=" * 70)

print(f"""
✓ PROJECT COMPLETED SUCCESSFULLY!

Dataset Statistics:
  - Total Students: {len(df_processed)}
  - Features Used: {X.shape[1]}
  - Classes: 3 (Dropout, Graduate, Enrolled)
  - Training Set: {X_train_balanced.shape[0]} (balanced with SMOTE)
  - Test Set: {X_test.shape[0]}

Models Trained & Evaluated:
  1. Logistic Regression - F1: {comparison_df[comparison_df['Model']=='Logistic Regression']['F1-Score'].values[0]:.4f}
  2. Support Vector Machine - F1: {comparison_df[comparison_df['Model']=='SVM']['F1-Score'].values[0]:.4f}
  3. Random Forest - F1: {comparison_df[comparison_df['Model']=='Random Forest']['F1-Score'].values[0]:.4f}
  4. Multi-Layer Perceptron - F1: {comparison_df[comparison_df['Model']=='MLP']['F1-Score'].values[0]:.4f}

Best Model: {best_model} (F1-Score: {best_f1:.4f})

Visualizations Generated:
  ✓ Class distributions & SMOTE effects
  ✓ Feature distributions & correlations
  ✓ Outlier analysis
  ✓ Model performance comparisons
  ✓ Confusion matrices
  ✓ Feature importance rankings
  ✓ Learning curves (DL)
  ✓ ML vs DL comparative analysis

Analysis Completed:
  ✓ Exploratory Data Analysis (EDA)
  ✓ Preprocessing & Class Imbalance Handling
  ✓ Model Training & Hyperparameter Tuning
  ✓ Cross-Validation (5-fold for ML models)
  ✓ Early Stopping Implementation (DL model)
  ✓ Performance Metrics Calculation
  ✓ Error Analysis & Interpretation
  ✓ Feature Importance Extraction
  ✓ ML vs DL Comparative Study

Next Steps:
  1. Export this notebook as PDF for presentation
  2. Share findings with team members
  3. Prepare presentation slides with key insights
  4. Discuss deployment strategy for dropout prediction system
""")
```

- [ ] **Step 6: Run final summary cell**

Expected Output shows comprehensive project summary.

- [ ] **Step 7: Final save and export**

Cell Type: Code
```python
print("\n✓ Notebook ready for export!")
print("\nTo export as PDF:")
print("1. File → Print (or Ctrl+P)")
print("2. Select 'Save as PDF'")
print("3. Save with filename: student_dropout_prediction_report.pdf")
```

- [ ] **Step 8: Save final notebook**

File → Save (Ctrl+S)
File → Download (.ipynb file to local)

---

### Task 13: Final Review & Quality Assurance

**Deliverable:** Verified working notebook ready for submission

- [ ] **Step 1: Run entire notebook end-to-end**

Runtime → Run all cells
Verify: All cells execute without errors
Expected: ~10-15 minutes total execution time

- [ ] **Step 2: Verify all visualizations are present**

Checklist:
- [ ] 4 class distribution plots (original, %, train original, train balanced)
- [ ] 12 feature distribution histograms
- [ ] 1 correlation heatmap
- [ ] 2 boxplots (outliers)
- [ ] 4 confusion matrices
- [ ] 1 feature importance bar chart (RF)
- [ ] 2 learning curves (loss & accuracy)
- [ ] 2 feature importance comparison plots
- [ ] 1 model performance comparison chart

- [ ] **Step 3: Verify all metrics are calculated and displayed**

Checklist:
- [ ] Accuracy, Precision, Recall, F1-Score for all 4 models
- [ ] Classification reports with per-class metrics
- [ ] Confusion matrices with correct dimensions (3x3 for 3 classes)
- [ ] Feature importance values
- [ ] ML vs DL comparison metrics

- [ ] **Step 4: Check for errors and warnings**

Runtime → Clear all outputs
Runtime → Run all cells
Verify: No error messages or warnings appear

- [ ] **Step 5: Download final notebook**

File → Download (.ipynb)
Saved as: `student_dropout_prediction.ipynb`

- [ ] **Step 6: Export as PDF report**

File → Print (Ctrl+P)
Print to → Save as PDF
Filename: `student_dropout_prediction_report.pdf`

- [ ] **Step 7: Final verification checklist**

```
✓ Notebook executes end-to-end without errors
✓ All 9 visualization categories present
✓ All metrics correctly calculated
✓ ML and DL models properly evaluated
✓ Feature importance extracted
✓ ML vs DL comparative analysis complete
✓ Final summary and insights included
✓ PDF report generated
✓ Code is well-commented and organized
✓ Ready for team submission and presentation
```

---

## Execution Approach

**Two options available:**

### **Option 1: Subagent-Driven (Recommended)** 
- Fresh subagent per task, review between tasks
- Faster iteration, independent verification
- Use: `superpowers:subagent-driven-development`

### **Option 2: Inline Execution**
- Execute tasks in this session with checkpoints
- Batch execution with reviews
- Use: `superpowers:executing-plans`

**Which approach do you prefer?**

---

**Plan saved to:** `docs/superpowers/plans/2026-05-11-student-dropout-ml-dl-comparison.md`
