# 📚 머신러닝 및 딥러닝 기반 대학생 중도포기 예측 프로젝트

> **인공지능학개론 팀과제 (11팀)**  
> **팀원:** 박재우, 염지훈, 오형우  
> **대상 파일:** [student_dropout_prediction.ipynb](file:///c:/Users/yumji/Desktop/학교/인공지능학%20개론/과제/팀과제/student_dropout_prediction.ipynb)

본 프로젝트는 대학생의 인구통계학적 배경, 사회경제적 환경, 학업 성취도 데이터를 분석하여 학생의 **중도포기(Dropout) 여부**를 선제적으로 예측하는 예측 알고리즘 모델(머신러닝 및 딥러닝)을 구축하고 성능을 비교한 과제입니다.

---

## 1. 📖 소스코드 및 분석 개요

본 소스코드(`student_dropout_prediction.ipynb`)는 정형 데이터를 활용한 분류 예측 문제 해결 과정을 담고 있습니다.

* **데이터셋 정보:** Kaggle Open Dataset (Predict students' dropout and academic success)
  * **데이터 규모:** 4,424행 × 35열 (34개 특징 피처, 1개 타겟 필드)
* **예측 대상 (Target):**
  * `1` (중도포기, Dropout): 1,421명
  * `0` (비중도포기, Non-Dropout - 졸업생 및 재학생 포함): 3,003명
* **핵심 처리 기법:**
  * **SMOTE (Synthetic Minority Over-sampling Technique):** 클래스 비율 불균형(중도포기 학생 비율 약 32%) 문제를 해결하기 위해 학습 데이터를 오버샘플링하여 비율을 50:50으로 맞춥니다.
  * **주성분 분석 (PCA):** 고차원의 다양한 피처 정보를 2차원으로 차원 축소하여 데이터 분포 패턴과 클래스별 경계를 시각적으로 분석합니다.
  * **하이퍼파라미터 튜닝:** `GridSearchCV` 기법과 5-Fold 교차검증을 연계하여 알고리즘별로 최적의 파라미터 조합을 탐색합니다.
  * **딥러닝 설계:** PyTorch 프레임워크를 기반으로 3개의 Linear Layer와 ReLU 활성화 함수, Dropout 기법을 적용한 신경망(MLP) 아키텍처를 구현했습니다.

---

## 2. 💻 주요 소스코드 셀 설명

노트북은 실행 흐름 순서대로 다음 5개의 단계로 이루어져 있습니다.

### Section 1. Setup & Data Loading
* 데이터 로드 및 전처리에 필요한 라이브러리(Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, PyTorch 등)를 임포트합니다.
* 코랩(Colab) 환경과 로컬 환경을 자동으로 식별하여 `dataset.csv` 파일을 자동으로 로딩하도록 설계되었습니다.

### Section 2. Data Preprocessing (데이터 전처리)
* **바이너리 타겟 설정:** 'Target' 변수의 'Dropout'을 `1`, 'Graduate'와 'Enrolled'를 `0`으로 인코딩하여 이진 분류 레이블을 형성합니다.
* **범주형 데이터 변환:** 문자 형태로 되어 있는 범주형 변수를 모델이 처리할 수 있게 `LabelEncoder`로 수치화합니다.
* **데이터 정규화:** 각 피처의 스케일 차이가 학습에 미치는 영향을 방지하기 위해 `StandardScaler`를 사용해 평균 0, 표준편차 1 상태로 정규화합니다.
* **학습/테스트 데이터 분할:** 전체 데이터를 7:3 비율로 나누며, 균등 분할을 위해 층화 추출(`stratify`)을 적용합니다.
* **SMOTE 오버샘플링:** 학습 데이터셋 내 소수 클래스인 Dropout 데이터를 늘려 데이터 수의 불균형을 극복합니다 (학습 데이터: 3,096개 $\rightarrow$ 4,204개로 균형 완료).

### Section 3. Exploratory Data Analysis (EDA)
* SMOTE 적용 전후의 클래스 분포 비율 변화를 막대 그래프로 시각화합니다.
* 주요 수치형 변수들의 분포 경향(히스토그램)을 파악합니다.
* PCA를 수행하여 다차원 정보를 2차원 평면에 시각화하여 특징 분포 경향성을 도식화합니다.

### Section 4. Model Development (모델 개발 및 학습)
* **Logistic Regression:** 선형 경계를 학습하는 모델로, 규제 강도 파라미터 `C`를 교차검증하여 최적의 파라미터로 학습합니다.
* **SVM (Support Vector Machine):** RBF(방사형 기저 함수) 커널을 기반으로 하여 비선형 패턴을 분류하며 최적의 `C`와 `gamma` 값을 튜닝합니다.
* **Random Forest:** 여러 개의 결정 트리를 조합해 다수결 투표를 하는 앙상블 모델로, 트리의 깊이와 분할 샘플 수 등을 최적화합니다.
* **PyTorch MLP:** `nn.Sequential`을 사용하여 3층 다층 퍼셉트론 아키텍처(은닉 노드: 128 $\rightarrow$ 64 $\rightarrow$ 32)를 구성하고 Dropout 레이어로 과적합을 제어하여 50 epoch 동안 미니배치 학습 루프를 직접 수행합니다.

### Section 5. Results Analysis (결과 분석 및 평가)
* 과제 2에서 자체 구현한 NumPy 기반의 `ClassificationMetrics` 클래스를 활용해 정확도, 정밀도, 재현율, F1-Score를 독자적으로 산출하여 모델을 다각도로 평가합니다.
* 각 분류 알고리즘의 오차 행렬(Confusion Matrix)을 Seaborn을 통해 2x2 히트맵 격자 형태로 그려 예측 실패 경향을 상세히 비교 분석합니다.

---

## 3. 🚀 실행 방법

### 사전 준비 사항
실행할 컴퓨터 환경에 파이썬이 준비되어 있어야 하며, 소스코드 파일(`student_dropout_prediction.ipynb`)과 원본 데이터 파일(`dataset.csv`)이 **반드시 동일한 폴더(디렉토리)**에 존재해야 합니다.

### 실행 절차 (로컬 PC 환경)
1. **터미널(Cmd 또는 PowerShell)을 실행하고 필요한 라이브러리를 설치합니다.**
   ```bash
   pip install pandas numpy scikit-learn imblearn matplotlib seaborn torch
   ```
2. **주피터 노트북 환경을 실행합니다.**
   ```bash
   jupyter notebook
   ```
3. **브라우저에 주피터 화면이 표시되면 `student_dropout_prediction.ipynb` 파일을 엽니다.**
4. **상단 메뉴에서 "Kernel" -> "Restart & Run All"을 선택하여 모든 코드 셀을 순차적으로 실행합니다.**

*(참고: Google Colab에서 실행하실 경우에는 Section 1.2의 파일 업로드 기능이 동작하므로, 선택 창이 나왔을 때 내 컴퓨터에 저장된 `dataset.csv`를 파일 다이얼로그를 통해 수동으로 업로드해주시면 됩니다.)*

---

## 📊 4. 최종 예측 결과 및 결론

테스트 데이터 1,328개에 대한 각 예측 모델별 최종 예측 성능 측정 결과입니다.

| 분류 알고리즘 (Model) | 정확도 (Accuracy) | 정밀도 (Precision) | 재현율 (Recall) | F1-Score (중도포기 클래스 1) |
| :--- | :---: | :---: | :---: | :---: |
| 🤖 **SVM (RBF Kernel)** | **87.88%** | **80.09%** | **82.90%** | **0.8147 (최고 성능)** |
| 📈 Logistic Regression | 87.42% | 77.66% | 85.48% | 0.8138 |
| 🧠 PyTorch MLP (신경망) | 87.88% | 81.37% | 80.80% | 0.8108 |
| 🌳 Random Forest | 87.58% | 81.19% | 79.86% | 0.8052 |

* **종합 결론:** 
  본 분석 태스크에서는 **SVM (RBF 커널)** 모델이 중도포기 클래스 예측 기준 F1-Score **0.8147**로 가장 높고 균형 잡힌 검출 능력을 발휘하였습니다. 수치형 피처가 잘 정형화된 데이터셋 특성상 데이터의 전체 개수(4,424개)가 비교적 작기 때문에, 복잡한 인공신경망 딥러닝 모델(MLP)보다는 적절히 최적화 튜닝된 머신러닝 전통 알고리즘들이 동등하거나 약간 우수한 탐지 효과를 보여주었습니다.
