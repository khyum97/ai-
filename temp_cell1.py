# ===== 섹션 5 Part B: 특징 중요도 분석 및 ML vs DL 비교 분석 =====
# 목표: 모델의 특징 중요도를 추출하고 ML과 DL의 차이점을 분석
# Part B: 특징 중요도 시각화, ML vs DL 비교표, 인사이트 제공

print("\n" + "=" * 70)
print("섹션 5 Part B: 특징 중요도 & ML vs DL 비교 분석")
print("=" * 70)

# ===== 5.5: 특징 중요도 추출 및 시각화 =====
# 특징 중요도란?
# - 각 특징(피처)이 모델의 예측에 얼마나 기여하는지를 나타내는 지표
# - Random Forest: 각 노드에서의 불순도(impurity) 감소로 측정
# - Logistic Regression: 계수(coefficient)의 절댓값으로 측정
# - 특징 중요도를 통해 학생 이탈에 가장 큰 영향을 미치는 요인 파악 가능

print("\n" + "=" * 50)
print("1. 특징 중요도 분석 (Feature Importance Analysis)")
print("=" * 50)

# ✓ 특징 이름 가져오기 (X.columns 사용)
# X는 전처리 과정에서 생성된 DataFrame으로, 모든 특징을 포함
feature_names = X.columns.tolist()

print(f"\n✓ 분석 대상 특징:")
print(f"  총 특징 수: {len(feature_names)}")
print(f"  특징 목록: {feature_names[:10]}... (처음 10개만 표시)")

# ===== 5.5.1: Random Forest 특징 중요도 추출 =====
# Random Forest의 feature_importances_는 트리 기반 앙상블 방식의 특징 중요도
# - 각 특징이 불순도(Gini index 또는 Entropy)를 얼마나 감소시키는지 측정
# - 값이 클수록: 모델의 예측에 더 많이 기여함
# - 모든 특징 중요도의 합은 1

print("\n" + "-" * 70)
print("Random Forest - 특징 중요도")
print("-" * 70)

# ✓ Random Forest 모델에서 특징 중요도 추출
rf_model_best = models_results['Random Forest']['model']
rf_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model_best.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n✓ 상위 15개 중요 특징 (Random Forest):")
print(rf_feature_importance.head(15).to_string(index=False))

print(f"\n✓ 특징 중요도의 의미:")
print(f"  - Importance > 0.1: 매우 중요한 특징 (모델 예측에 크게 영향)")
print(f"  - Importance 0.05~0.1: 중요한 특징")
print(f"  - Importance < 0.05: 상대적으로 덜 중요한 특징")
print(f"  - 학생 이탈 예측에서 높은 중요도 특징은 실제 이탈 원인과 일치하는지 검토 필요")

# ===== 5.5.2: Logistic Regression 계수 추출 =====
# Logistic Regression의 계수(coef_)는 각 특징의 선형 영향도를 나타냄
# - 양수: 그 특징이 증가하면 목표 클래스 확률 증가
# - 음수: 그 특징이 증가하면 목표 클래스 확률 감소
# - 절댓값이 클수록: 더 강한 영향력
# - 선형 모델이므로 특징 간의 상호작용(interaction)은 포착 불가

print("\n" + "-" * 70)
print("Logistic Regression - 특징 계수 (절댓값)")
print("-" * 70)

# ✓ Logistic Regression 모델에서 계수 추출
lr_model_best = models_results['Logistic Regression']['model']

# 다중 클래스 분류의 경우 각 클래스별 계수를 평균
lr_coef = np.abs(lr_model_best.coef_).mean(axis=0)

lr_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_coef
}).sort_values('Coefficient', ascending=False)

print(f"\n✓ 상위 15개 영향 특징 (Logistic Regression):")
print(lr_feature_importance.head(15).to_string(index=False))

print(f"\n✓ 로지스틱 회귀 계수의 의미:")
print(f"  - 선형 모델이므로 각 특징의 영향이 직선적(linear)임")
print(f"  - 실제 계수 값(부호)을 통해 양/음의 영향 파악 가능")
print(f"  - 하지만 다중 클래스 분류에서는 각 클래스별 계수가 다르므로, 평균으로 표시")

# ===== 5.5.3: 특징 중요도 시각화 (Side-by-side 비교) =====
# Random Forest와 Logistic Regression의 특징 중요도를 나란히 비교
# - RF: 비선형 관계와 특징 간의 상호작용 포착 가능
# - LR: 선형 관계만 포착

print("\n" + "-" * 70)
print("특징 중요도 시각화 (RF vs LR)")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ✓ Random Forest 특징 중요도 시각화
ax = axes[0]
top_rf = rf_feature_importance.head(15)
colors_rf = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_rf)))
ax.barh(range(len(top_rf)), top_rf['Importance'].values, color=colors_rf, edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(top_rf)))
ax.set_yticklabels(top_rf['Feature'].values, fontsize=10)
ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Features - Random Forest\n(비선형 + 특징 상호작용)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# ✓ Logistic Regression 계수 시각화
ax = axes[1]
top_lr = lr_feature_importance.head(15)
colors_lr = plt.cm.Oranges(np.linspace(0.4, 0.8, len(top_lr)))
ax.barh(range(len(top_lr)), top_lr['Coefficient'].values, color=colors_lr, edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(top_lr)))
ax.set_yticklabels(top_lr['Feature'].values, fontsize=10)
ax.set_xlabel('|Coefficient| (Absolute Value)', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Features - Logistic Regression\n(선형 관계)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ 특징 중요도 시각화 완료")

# ===== 5.5.4: 공통 중요 특징 식별 =====
# RF와 LR 모두에서 상위 특징으로 나타나는 특징들 찾기
# 이는 여러 모델이 일치하는 특징 = 실제로 중요할 가능성이 높음

print("\n" + "-" * 70)
print("공통 중요 특징 분석 (RF & LR 합의)")
print("-" * 70)

rf_top10 = set(rf_feature_importance.head(10)['Feature'])
lr_top10 = set(lr_feature_importance.head(10)['Feature'])
common_features = rf_top10.intersection(lr_top10)

print(f"\n✓ Random Forest 상위 10개 특징:")
print(f"  {list(rf_top10)}")

print(f"\n✓ Logistic Regression 상위 10개 특징:")
print(f"  {list(lr_top10)}")

print(f"\n✓ 두 모델 모두에서 상위 10위의 공통 특징 ({len(common_features)}개):")
if common_features:
    for i, feat in enumerate(sorted(list(common_features)), 1):
        rf_rank = rf_feature_importance[rf_feature_importance['Feature'] == feat].index[0] + 1
        lr_rank = lr_feature_importance[lr_feature_importance['Feature'] == feat].index[0] + 1
        print(f"  {i}. {feat:20s} - RF순위: {rf_rank:2d}, LR순위: {lr_rank:2d}")
else:
    print("  (공통 특징이 없음 - 두 모델이 다른 패턴을 학습)")

# ===== 5.6: ML vs DL 비교 분석 =====
# 머신러닝(ML)과 딥러닝(DL)의 근본적인 차이 분석
# ML: 인간이 설계한 특징 사용, 선형/비선형 경계 학습
# DL: 데이터로부터 특징 자동 학습, 깊은 표현력 학습

print("\n" + "=" * 50)
print("2. ML vs DL 비교 분석")
print("=" * 50)

# ===== 5.6.1: 모델 비교 표 생성 =====
# 4개 모델의 특성과 성능을 한눈에 비교

print("\n" + "-" * 70)
print("모든 모델 비교표")
print("-" * 70)

# ✓ 비교 데이터 구성
comparison_data = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'MLP (Neural Network)'],
    'Type': ['ML (선형)', 'ML (비선형)', 'ML (앙상블)', 'DL (딥러닝)'],
    'Architecture/Method': [
        '선형 확률 모델',
        'Kernel-based (RBF)',
        '100개 의사결정트리',
        '3개 은닉층 (128-64-32)'
    ],
    'Accuracy': [
        models_results['Logistic Regression']['accuracy'],
        models_results['SVM']['accuracy'],
        models_results['Random Forest']['accuracy'],
        models_results['MLP']['accuracy']
    ],
    'Precision': [
        models_results['Logistic Regression']['precision'],
        models_results['SVM']['precision'],
        models_results['Random Forest']['precision'],
        models_results['MLP']['precision']
    ],
    'Recall': [
        models_results['Logistic Regression']['recall'],
        models_results['SVM']['recall'],
        models_results['Random Forest']['recall'],
        models_results['MLP']['recall']
    ],
    'F1-Score': [
        models_results['Logistic Regression']['f1'],
        models_results['SVM']['f1'],
        models_results['Random Forest']['f1'],
        models_results['MLP']['f1']
    ],
    'Interpretability': ['매우 높음', '낮음', '높음', '매우 낮음'],
    'Complexity': ['낮음', '중간', '중간', '높음'],
    'Training Speed': ['매우 빠름', '중간', '빠름', '느림']
}

comparison_df = pd.DataFrame(comparison_data)

print("\n✓ 모델 종합 비교표:")
print(comparison_df.to_string(index=False))

# ===== 5.6.2: 성능 메트릭 상세 분석 =====
print("\n" + "-" * 70)
print("성능 메트릭 분석")
print("-" * 70)

# 각 지표별 최고 성능 모델 강조
print("\n✓ 각 지표별 최고 성능 모델:")
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for metric in metrics_list:
    best_idx = comparison_df[metric].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_score = comparison_df.loc[best_idx, metric]
    
    # ML vs DL 구분
    model_type = "ML" if best_idx < 3 else "DL"
    print(f"  {metric:12s}: {best_model:25s} ({best_score:.4f}) [{model_type}]")

# ===== 5.6.3: ML vs DL의 근본적인 차이 설명 =====
print("\n" + "-" * 70)
print("ML vs DL 근본적 차이점 분석")
print("-" * 70)

ml_vs_dl = """
✓ ML (머신러닝) vs DL (딥러닝)의 근본적 차이:

1. 특징 표현 (Feature Representation)
   ┌─ 머신러닝 (ML)
   │  - 인간이 직접 설계한 특징(handcrafted features) 사용
   │  - 예: Random Forest는 주어진 특징을 그대로 사용
   │  - 특징이 제대로 설계되면 성능 우수, 아니면 성능 저하
   │
   └─ 딥러닝 (DL)
      - 데이터로부터 자동으로 특징 학습 (learned representations)
      - 각 레이어에서 복잡한 패턴을 자동 발견
      - 특징 엔지니어링 불필요, 적응적 학습

2. 비선형성 처리 능력
   ┌─ 머신러닝
   │  - Logistic Regression: 선형 경계만 학습 (제한적)
   │  - SVM (RBF kernel): 커널 트릭으로 비선형 경계 학습
   │  - Random Forest: 트리 구조로 비선형 관계 포착
   │  - 하지만 매우 복잡한 패턴은 어려움
   │
   └─ 딥러닝
      - 다층 신경망으로 매우 복잡한 비선형 함수 근사 가능
      - 각 은닉층이 더 추상적인 표현으로 변환
      - 이론상 임의의 복잡도 함수 표현 가능

3. 해석성 (Interpretability)
   ┌─ 머신러닝
   │  - Logistic Regression: 각 특징의 계수로 영향 파악 가능
   │  - Random Forest: 특징 중요도로 어느 특징이 중요한지 알 수 있음
   │  - "왜" 이 결정을 했는지 설명 가능 → 실무에서 신뢰도 높음
   │
   └─ 딥러닝
      - 신경망의 내부 구조 파악 어려움 ("블랙박스")
      - 특징 중요도 추출 불가 (각 레이어가 특징을 추상적으로 변환)
      - 최근 XAI(설명 가능한 AI) 기법 연구 중이지만 아직 미흡

4. 데이터와 계산 자원 요구
   ┌─ 머신러닝
   │  - 적은 데이터로도 우수한 성능 가능 (수백~수천 샘플)
   │  - 계산 자원 요구 적음 (CPU만으로도 충분)
   │  - 학습 속도 빠름
   │
   └─ 딥러닝
      - 대량의 데이터 필요 (수만~수백만 샘플)
      - 높은 계산 자원 필요 (GPU/TPU 권장)
      - 학습 시간 김 (하지만 병렬 처리로 완화 가능)

5. 과적합 (Overfitting) 경향
   ┌─ 머신러닝
   │  - 상대적으로 과적합 적음
   │  - 모델 복잡도 제어 용이 (max_depth, regularization C 등)
   │
   └─ 딥러닝
      - 과적합 경향 높음 (파라미터가 많기 때문)
      - 더 많은 정규화 기법 필요 (Dropout, Early Stopping, L1/L2)

✓ 이 프로젝트 (학생 이탈 예측)에서의 implications:
   - 데이터셋 크기: 약 2,000명 (상대적으로 작음)
   - ML 모델이 적합한 선택: 적은 데이터로도 설명 가능한 예측
   - DL이 크게 우위를 보이기 어려움: 충분한 데이터가 없음
   - 현실적 적용: 특징 중요도가 파악되는 RF/LR 이 더 신뢰도 높음
"""

print(ml_vs_dl)

# ===== 5.7: 핵심 인사이트 및 권장사항 =====
print("\n" + "=" * 50)
print("3. 핵심 인사이트 및 실무 권장사항")
print("=" * 50)

insights = """
✓ 특징 중요도 분석의 교육적 의의:

1. 학생 이탈의 주요 영향 요인 규명
   - Random Forest가 식별한 상위 특징들이 실제 이탈 원인과 일치하는가?
   - 데이터 기반의 객관적 증거로 대학의 학생지원 정책 수립 가능
   - 예: "attendance가 가장 중요한 특징이다" → 수강 관리 강화 권장

2. ML vs DL 선택 기준 (일반적인 가이드라인):
   
   ┌─ 머신러닝 선택하기 (이 프로젝트의 경우)
   │  ✓ 데이터가 적을 때 (< 10,000 샘플)
   │  ✓ 해석 가능성이 중요할 때 (의사 결정 근거 필요)
   │  ✓ 계산 자원 제약이 있을 때
   │  ✓ 실시간 예측이 필요할 때 (빠른 학습과 예측)
   │  → Logistic Regression (빠름), Random Forest (정확함)
   │
   └─ 딥러닝 선택하기
      ✓ 대용량 데이터가 있을 때 (> 100,000 샘플)
      ✓ 구조화되지 않은 데이터 (이미지, 음성, 텍스트)
      ✓ 매우 복잡한 패턴이 있을 때
      ✓ 최고 성능이 필수적일 때 (비용이 문제 아닐 때)
      ✓ 사전 학습된 모델(transfer learning) 활용 가능할 때

3. 이 프로젝트의 결론:
   
   우선순위:
   1순위: Random Forest - 높은 정확도 + 특징 중요도 파악 가능
   2순위: Logistic Regression - 빠른 학습 + 선형 해석 + 배포 용이
   3순위: SVM - 좋은 성능 + 중간 정도의 복잡도
   4순위: MLP - 이 규모의 데이터에선 과잉 설계 (overfitting 위험)

4. 실무 배포 시 체크리스트:
   □ 특징 중요도를 대학 관계자와 함께 검토 (도메인 전문가 의견 중요)
   □ Random Forest의 특징 중요도를 토대로 개입 전략 수립
   □ 주기적으로 모델 성능 모니터링 (데이터 드리프트 감지)
   □ 불균형 처리 (SMOTE)의 실제 필요성 재평가
   □ 새로운 학년 데이터로 모델 재학습

5. 향후 개선 방향:
   □ 더 많은 데이터 수집 → DL 모델의 잠재력 발휘 가능
   □ 특징 엔지니어링: 시간 관련 특징 추가 (예: 주별 attendance 추이)
   □ 앙상블 방법: Random Forest + Logistic Regression 결합
   □ 클래스별 최적화: Dropout, Graduate, Enrolled 각각 다른 모델 사용
   □ 설명 가능성 강화: SHAP, LIME 등 XAI 기법 적용
"""

print(insights)

# ===== 5.8: 모델 예측 신뢰도 시각화 =====
# 각 모델의 예측이 얼마나 일치하는지 시각화

print("\n" + "=" * 50)
print("4. 모델 간 예측 일치도 분석")
print("=" * 50)

# ✓ 예측 결과 비교
predictions_dict = {
    'Logistic Regression': models_results['Logistic Regression']['predictions'],
    'SVM': models_results['SVM']['predictions'],
    'Random Forest': models_results['Random Forest']['predictions'],
    'MLP': models_results['MLP']['predictions']
}

# 4개 모델이 모두 같은 예측을 한 경우의 비율
all_models_agree = np.sum(
    (predictions_dict['Logistic Regression'] == predictions_dict['SVM']) &
    (predictions_dict['SVM'] == predictions_dict['Random Forest']) &
    (predictions_dict['Random Forest'] == predictions_dict['MLP'])
) / len(y_test) * 100

print(f"\n✓ 모델 예측 일치도:")
print(f"  4개 모델이 모두 같은 예측: {all_models_agree:.1f}%")
print(f"  → {all_models_agree:.1f}%의 샘플에서 모든 모델이 일치")

# ML 모델들의 일치도
ml_agree = np.sum(
    (predictions_dict['Logistic Regression'] == predictions_dict['SVM']) &
    (predictions_dict['SVM'] == predictions_dict['Random Forest'])
) / len(y_test) * 100

print(f"  ML 모델 3개의 일치도: {ml_agree:.1f}%")
print(f"  → 이 비율이 높을수록 예측이 안정적")

# 다수결 투표로 최종 예측
from scipy.stats import mode

all_preds = np.column_stack([
    predictions_dict['Logistic Regression'],
    predictions_dict['SVM'],
    predictions_dict['Random Forest'],
    predictions_dict['MLP']
])

ensemble_pred, _ = mode(all_preds, axis=1, keepdims=False)

# 앙상블 성능 평가
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

print(f"\n✓ 앙상블 (4개 모델 투표) 성능:")
print(f"  Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"  Ensemble F1-Score: {ensemble_f1:.4f}")

# 각 개별 모델과 비교
print(f"\n개별 모델과의 비교:")
for model_name in predictions_dict.keys():
    individual_f1 = models_results[model_name]['f1']
    improvement = (ensemble_f1 - individual_f1) * 100
    symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
    print(f"  {model_name:25s}: {individual_f1:.4f} → {ensemble_f1:.4f} {symbol} ({improvement:+.1f}%)")

print("\n✓ 해석:")
print("  - 여러 모델의 투표 결과가 개별 모델보다 나을 수 있음")
print("  - 하지만 모델이 모두 같은 방향으로 편향되면 효과 제한적")
print("  - 다양한 아키텍처의 모델 조합이 가장 효과적")

print("\n" + "=" * 70)
print("✓ Section 5 Part B 완료!")
print("=" * 70)
print("""
✓ 분석 요약:
  1. 특징 중요도: Random Forest와 Logistic Regression 비교
  2. ML vs DL: 4개 모델의 특성과 성능 비교
  3. 핵심 인사이트: 학생 이탈 예측의 주요 영향 요인 도출
  4. 실무 권장사항: 모델 선택 및 배포 가이드라인

✓ 이후 단계:
  - 특징 중요도 결과를 대학의 학생지원 정책과 연계
  - Random Forest를 배포 모델로 선정하고 모니터링 체계 구축
  - 정기적인 모델 재학습 스케줄 수립
  - 예측 결과의 설명 가능성 강화 (XAI 기법 추가)
""")

print("\n" + "=" * 70)
print("✓ 전체 프로젝트 완료!")
print("=" * 70)