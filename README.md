# 📚 학생 중도포기 예측: ML vs DL 비교 분석 (Dropout Prediction Only)

> **팀 11 - 인공지능학 개론 최종 프로젝트**  
> 팀원: 박재우 (리더), 염지훈, 오형우  
> 대상 파일: [student_dropout_prediction.ipynb](file:///c:/Users/yumji/Desktop/학교/인공지능학%20개론/과제/팀과제/student_dropout_prediction.ipynb)

---

## 📋 프로젝트 개요 (Project Overview)

### 🎯 목표
대학생의 인구통계학적 배경, 사회경제적 요인, 학업 성취도 데이터를 활용하여 학생의 **중도포기(Dropout)** 여부를 선제적으로 예측하는 머신러닝(ML) 및 딥러닝(DL) 모델을 개발하고 성능을 비교 분석합니다.

### 📊 데이터셋 및 타겟 정의
- **데이터셋:** Kaggle - Predict students' dropout and academic success
- **크기:** 4,424명 학생 × 35개 특징 (Features)
- **타겟 변수 (Target):**
  - `y = 1`: **중도포기 (Dropout)** — 1,421명 (32.1%)
  - `y = 0`: **비중도포기 (Non-Dropout)** — 3,003명 (67.9%) — [졸업생 `Graduate`(2,209명) + 등록생 `Enrolled`(794명)]
- **클래스 불균형 해결:** 학습 세트의 비율 불균형을 극복하기 위해 **SMOTE (Synthetic Minority Over-sampling Technique)** 기법을 활용하여 학습 데이터를 3,096개에서 4,204개로 오버샘플링하여 클래스 비율을 50:50으로 균형 있게 조정했습니다.

---

## 📈 모델링 결과 및 성능 비교 (Key Results)

4가지 알고리즘(Logistic Regression, SVM, Random Forest, PyTorch MLP)을 사용하여 훈련 및 튜닝한 최종 결과 성능 비교 테이블입니다. (테스트 데이터 1,328개에 대한 평가 지표)

| 모델 (Model) | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🤖 **SVM (RBF Kernel)** | **87.88%** | **80.09%** | **82.90%** | **0.8147** | ⭐ **최고 성능 (Best)** |
| 🧠 PyTorch MLP (DL) | 87.88% | 81.37% | 80.80% | 0.8108 | 딥러닝 비교 모델 |
| 📈 Logistic Regression | 87.42% | 77.66% | 85.48% | 0.8138 | 선형 기준선 모델 |
| 🌳 Random Forest | 87.58% | 81.19% | 79.86% | 0.8052 | 앙상블 비교 모델 |

### 🔑 주요 발견사항
1. **최적 모델 선정:** **SVM (RBF Kernel)** 모델이 F1-Score **0.8147**로 중도포기 예측 태스크에서 가장 우수한 성능을 보여 최종 배포 모델로 권장됩니다.
2. **ML vs DL 비교:** 정형 데이터셋의 한계 및 데이터 규모(4,424개) 특성상 전통적인 머신러닝 알고리즘(SVM, Logistic Regression)이 PyTorch 기반 딥러닝 모델(MLP)에 비해 학습 속도 및 예측 성능 면에서 **약 0.1% ~ 0.4%** 우수한 성능을 보여주었습니다.
3. **핵심 예측 요인:** Random Forest 특징 중요도 분석 결과, 첫 학기/둘째 학기 학점 성적과 함께 **학비 납부 현황(Tuition fees up to date)** 및 **장학금 수혜 여부(Scholarship holder)**가 학생의 중도포기 여부를 결정하는 핵심 사회경제적 지표로 분석되었습니다.

---

## 🚀 빠른 시작 (Quick Start)

### 💻 로컬 Jupyter Notebook 실행
Jupyter 환경에서 `dataset.csv` 파일이 동일한 폴더에 위치해 있는지 확인하고 노트북을 실행합니다.

```bash
# 필수 라이브러리 설치
pip install pandas numpy scikit-learn imblearn matplotlib seaborn torch

# 주피터 노트북 실행
jupyter notebook student_dropout_prediction.ipynb
```

---

## 📖 노트북 구성 (Notebook Structure)

- **Section 1: Setup & Data Loading** — 필수 라이브러리 로드 및 dataset.csv 로딩
- **Section 2: Data Preprocessing** — 수치형 변수 정규화(`StandardScaler`), 범주형 변수 인코딩, `SMOTE` 불균형 해소
- **Section 3: Exploratory Data Analysis (EDA)** — 클래스 분포 시각화 및 PCA 주성분 분석 기반 2차원 투영 시각화
- **Section 4: Model Development** — `GridSearchCV` 하이퍼파라미터 튜닝 기반 머신러닝 모델 학습 및 PyTorch 기반 3층 MLP 설계/학습
- **Section 5: Results Analysis** — 커스텀 `ClassificationMetrics` 평가 및 2x2 혼동행렬(Confusion Matrix) 그리드 시각화 비교

---

---

# 📚 Student Dropout Prediction: ML vs DL Comparison (Dropout Prediction Only)

> **Team 11 - AI Introduction Final Project**  
> Members: Park Jae-woo (Leader), Yum Ji-hun, Oh Hyung-woo  
> Target File: [student_dropout_prediction.ipynb](file:///c:/Users/yumji/Desktop/학교/인공지능학%20개론/과제/팀과제/student_dropout_prediction.ipynb)

---

## 📋 Project Overview

### 🎯 Goal
Develop and evaluate Machine Learning (ML) and Deep Learning (DL) models to proactively predict student **dropout** using demographic, socioeconomic, and academic performance datasets, helping universities identify at-risk students.

### 📊 Dataset and Target Definition
- **Dataset:** Kaggle - Predict students' dropout and academic success
- **Size:** 4,424 students × 35 features
- **Target Variable (Target):**
  - `y = 1`: **Dropout** — 1,421 students (32.1%)
  - `y = 0`: **Non-Dropout** — 3,003 students (67.9%) — [`Graduate` (2,209) + `Enrolled` (794)]
- **Handling Class Imbalance:** To address the class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied, oversampling the training data from 3,096 to 4,204 samples to achieve a balanced 50:50 ratio.

---

## 📈 Modeling Results & Performance Comparison

Comparison of the final performance evaluation metrics on the test set (1,328 samples).

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🤖 **SVM (RBF Kernel)** | **87.88%** | **80.09%** | **82.90%** | **0.8147** | ⭐ **Best Model** |
| 🧠 PyTorch MLP (DL) | 87.88% | 81.37% | 80.80% | 0.8108 | Deep Learning Model |
| 📈 Logistic Regression | 87.42% | 77.66% | 85.48% | 0.8138 | Linear Baseline Model |
| 🌳 Random Forest | 87.58% | 81.19% | 79.86% | 0.8052 | Ensemble Model |

### 🔑 Key Findings
1. **Best Model:** The **SVM (RBF Kernel)** model achieved the best target class F1-Score of **0.8147** and Accuracy of **87.88%**, making it the recommended model for deployment.
2. **ML vs DL Comparison:** On this tabular dataset, traditional ML models (SVM, Logistic Regression) slightly outperformed the PyTorch MLP model by **0.1% to 0.4%**. This is primarily because the dataset size (4,424 samples) is relatively small for deep learning to show its full potential.
3. **Core Predictors:** Feature importance from Random Forest shows that academic performance in the first and second semesters, **Tuition fees up to date**, and **Scholarship holder** status are the most critical predictors of student dropout.

---

## 🚀 Quick Start

### 💻 Running Locally
Ensure `dataset.csv` is placed in the same directory as the notebook.

```bash
# Install required libraries
pip install pandas numpy scikit-learn imblearn matplotlib seaborn torch

# Launch Jupyter Notebook
jupyter notebook student_dropout_prediction.ipynb
```
