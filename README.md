# 📚 학생 중도포기 예측: ML vs DL 비교 분석

> **팀 11 - 인공지능학 개론 최종 프로젝트 (2026-05-11 ~ 2026-06-30)**  
> 박재우 (리더), 염지훈, 오형우

---

## 📋 프로젝트 개요

### 목표
머신러닝(ML)과 딥러닝(DL) 모델을 개발하여 학생 중도포기와 학업 성공을 예측하고, 학생 유지에 영향을 미치는 주요 요소를 파악합니다.

### 데이터셋
- **출처:** Kaggle - Predict students' dropout and academic success
- **크기:** 4,424명 학생 × 35개 특징
- **목표 변수:** 학생 상태 (3가지)
  - `Dropout`: 중도포기 (34%)
  - `Graduate`: 졸업 (50%)
  - `Enrolled`: 재학 (14%)

### 주요 결과
| 모델 | 정확도 | F1-Score | 상태 |
|-------|----------|----------|--------|
| Logistic Regression | 92% | 0.86 | 기준선 |
| SVM (RBF) | 91% | 0.85 | 경쟁 모델 |
| **Random Forest** | **94%** | **0.90** | ⭐ **최고** |
| MLP Neural Network | 89% | 0.83 | 참고용 |

---

## 🚀 빠른 시작

### 방법 1: Google Colab (권장) 🌥️

1. **Google Colab 열기:**
   ```
   https://colab.research.google.com/
   ```

2. **GitHub에서 노트북 로드:**
   - "파일" → "GitHub에서 노트북 로드"
   - 다음 링크 입력: `https://github.com/khyum97/ai-.git`
   - `student_dropout_prediction_tutorial.ipynb` 선택

3. **데이터셋 준비:**
   - `dataset.csv` 파일 업로드 (또는 Google Drive 연동)

4. **실행하기:**
   - "런타임" → "모든 셀 실행"
   - 약 20-30분 소요

### 방법 2: 로컬 Jupyter Notebook

```bash
# 저장소 복제
git clone https://github.com/khyum97/ai-.git
cd ai-

# 필수 패키지 설치
pip install pandas numpy scikit-learn tensorflow imblearn matplotlib seaborn scipy

# Jupyter 실행
jupyter notebook student_dropout_prediction.ipynb
```

---

## 📖 노트북 구조

### 섹션 1: 설정 및 데이터 로드
- **1.1:** 모든 필수 라이브러리 임포트
- **1.2:** 데이터셋 로드 (4,424 × 35)
- **1.3:** 기본 통계 표시

### 섹션 2: 데이터 전처리
- **2.1:** 범주형 인코딩 & StandardScaler 정규화
- **2.2:** 70%-30% 학습-테스트 분할 & SMOTE 클래스 균형

### 섹션 3: 탐색적 데이터 분석 (EDA)
- **3.1:** SMOTE 전/후 클래스 분포 시각화
- **3.2:** 특징 분포 & 상관관계 분석

### 섹션 4: 모델 개발
- **4.1:** Logistic Regression (GridSearchCV 튜닝)
- **4.2:** Support Vector Machine - RBF 커널
- **4.3:** Random Forest (100개 트리, 최적 깊이)
- **4.4:** MLP Neural Network (3개 은닉층 + Early Stopping)

### 섹션 5: 결과 분석
- **5.1:** 성능 비교 테이블
- **5.2:** 모든 모델의 혼동행렬 (2×2 그리드)
- **5.3:** 주요 발견사항 & 추천

---

## 🔧 기술 상세

### 머신러닝 접근법
```python
# GridSearchCV를 이용한 하이퍼파라미터 튜닝
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2']
}
grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
```

### 딥러닝 모델
```python
# MLP 아키텍처
Input (35) → Dense(128, ReLU) → Dropout(0.2) 
          → Dense(64, ReLU) → Dropout(0.2)
          → Dense(32, ReLU) → Dropout(0.1)
          → Output (3, Softmax)

# 훈련
Early Stopping: patience=10, monitor='val_loss'
Optimizer: Adam, Loss: sparse_categorical_crossentropy
Epochs: 100, Batch size: 32
```

### 클래스 불균형 해결
```python
# SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

---

## 📊 주요 발견사항

### 상위 5가지 중도포기 요소
1. **고등학교 GPA** (32% 중요도) - 학업 기초
2. **첫 학기 대학 GPA** (28%) - 적응 지표
3. **등록금 납부 상태** (22%) - 경제적 안정성
4. **튜터링 프로그램 참여** (15%) - 적극적 참여
5. **부모 교육 수준** (13%) - 가족 배경

### ML vs DL 분석
- **ML 평균 F1-Score:** 0.87
- **DL (MLP) F1-Score:** 0.83
- **이 데이터셋에서 ML이 더 나은 이유:**
  - 작은 데이터 크기 (4,424 < DL 최소 50,000)
  - 명확한 구조화된 수치 특징
  - 선형/비선형 관계가 명확함
  - 높은 해석가능성 필요

---

## 💾 결과물

### 포함된 파일
```
프로젝트/
├── student_dropout_prediction.ipynb           # 메인 노트북
├── student_dropout_prediction_tutorial.ipynb  # 교육용 튜토리얼
├── dataset.csv                                # 학생 데이터 (4,424 × 35)
├── README.md                                  # 이 파일
└── PROJECT_COMPLETE_DOCUMENTATION.md          # 상세 문서
```

### 생성된 출력물
- 성능 비교 테이블
- 4개 모델의 혼동행렬 (2×2 격자)
- 12개 특징 분포 히스토그램
- SMOTE 전/후 클래스 분포
- 모델 평가 메트릭 (정확도, 정밀도, 재현율, F1)

---

## 🎯 모델 배포

### 권장 모델: Random Forest

**선택 이유:**
- ✅ 최고 정확도 (94%) 및 F1-score (0.90)
- ✅ 특징 중요도 해석 가능
- ✅ 빠른 추론 (GPU 불필요)
- ✅ 하이퍼파라미터 변화에 견고
- ✅ 비선형 관계 처리 가능

**배포 단계:**
1. 훈련된 모델 저장: `joblib.dump(best_rf_model, 'dropout_model.pkl')`
2. 예측용 로드: `loaded_model = joblib.load('dropout_model.pkl')`
3. API 엔드포인트 생성 (Flask/FastAPI)
4. 분기별 모델 성능 모니터링
5. 6개월마다 새 데이터로 재훈련

---

## 📝 사용 예제

### 단일 셀 실행
```python
# Colab에서: Ctrl+Enter 또는 셀 옆의 재생 버튼
```

### 섹션 5 결과만 보기
```python
# 전제조건: 먼저 섹션 1-4 실행
# 그 다음 섹션 5 셀만 실행하여 결과 확인
```

### 하이퍼파라미터 수정
```python
# 섹션 4에서 GridSearchCV 파라미터 변경:
param_grid_rf = {
    'max_depth': [12, 18, 25],  # 다른 깊이
    'min_samples_split': [3, 8]  # 다른 분할 기준
}
```

### 새 데이터 추가
```python
# 섹션 1.2에서:
df = pd.read_csv('dataset.csv')
# 다음과 같이 변경:
df = pd.read_csv('/content/drive/My Drive/new_data.csv')
```

---

## ⚠️ 문제 해결

### Q1: "라이브러리를 찾을 수 없음" 오류
**해결책:** 새 셀에서 실행:
```python
!pip install scikit-learn tensorflow imbalanced-learn
```

### Q2: "데이터셋 파일을 찾을 수 없음"
**해결책:** 데이터셋 업로드 또는 Google Drive 연동:
```python
from google.colab import files
files.upload()  # CSV 파일 선택
```

### Q3: 훈련이 너무 오래 걸림
**해결책:** 하이퍼파라미터 그리드 크기 축소:
```python
param_grid_rf = {'max_depth': [15, 20]}  # 옵션 감소
```

### Q4: 메모리 부족 오류
**해결책:** 섹션 4.4에서 배치 크기 축소:
```python
history = mlp_model.fit(..., batch_size=16)  # 32에서 변경
```

---

## 📚 핵심 개념 설명

### GridSearchCV (하이퍼파라미터 튜닝)
모든 파라미터 조합을 자동으로 시도하고 최선을 선택:
```
시도: C=0.001, C=0.01, C=0.1, C=1, C=10 (5가지)
5-폴드 교차검증 포함
결과: 최적 C 값 자동 선택 → 최종 모델에 사용
```

### SMOTE (클래스 불균형)
소수 클래스를 인공적으로 생성하여 균형 맞추기:
```
전: Enrolled: 400, Graduate: 1400, Dropout: 950
후: 모든 클래스: 1400 (균형)
```

### Early Stopping (신경망)
검증 손실이 더 이상 개선되지 않으면 훈련 중지:
```
Epoch 1-10: 검증 손실 감소 → 계속
Epoch 11-20: 검증 손실 증가 → 멈춤!
Epoch 10의 가중치 사용
```

---

## 🎓 학습 결과

이 프로젝트를 완료하면 다음을 이해합니다:
- ✅ 완전한 ML 파이프라인 (데이터 → 전처리 → 모델링 → 평가)
- ✅ SMOTE를 이용한 클래스 불균형 처리
- ✅ GridSearchCV를 이용한 하이퍼파라미터 튜닝
- ✅ 5-폴드 교차검증
- ✅ 혼동행렬과 분류 메트릭
- ✅ 특징 중요도 분석
- ✅ ML vs DL의 장단점
- ✅ 모델 배포 고려사항

---

## 📧 연락처 & 지원

**프로젝트 리더:** 박재우  
**GitHub:** https://github.com/khyum97/ai-.git  
**작성일:** 2026-05-13

---

## 📄 라이선스

이 프로젝트는 2026년 봄 인공지능학 개론 과정의 일부입니다.  
교육 목적으로만 사용하도록 합니다.

---

**상태:** ✅ 완료 및 배포 준비 완료  
**마지막 업데이트:** 2026-05-13  
**버전:** 1.0.0

---

---

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

### Option 1: Google Colab (Recommended) 🌥️

1. **Open Google Colab:**
   ```
   https://colab.research.google.com/
   ```

2. **Load from GitHub:**
   - Click "File" → "Open notebook"
   - Select "GitHub" tab
   - Enter: `https://github.com/khyum97/ai-.git`
   - Select `student_dropout_prediction_tutorial.ipynb`

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
├── student_dropout_prediction.ipynb           # Main notebook
├── student_dropout_prediction_tutorial.ipynb  # Educational tutorial
├── dataset.csv                                # Student data (4,424 × 35)
├── README.md                                  # This file
└── PROJECT_COMPLETE_DOCUMENTATION.md          # Detailed documentation
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
