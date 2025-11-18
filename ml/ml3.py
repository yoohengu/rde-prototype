"""
기업 부도 예측 모델 비교: 원본 데이터 vs SMOTE 증강 데이터
- 동일한 알고리즘 3가지 사용 (Logistic Regression, Random Forest, Gradient Boosting)
- 모델 A: 원본 데이터 + class_weight 조정
- 모델 B: SMOTE 증강 데이터
"""
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("기업 부도 예측 모델 비교: 원본 데이터 vs SMOTE 증강 데이터")
print("=" * 100)

# ==================== 1. 데이터 로드 ====================
print("\n[1] 데이터 로드 중...")
load_dotenv()
df_path = os.getenv("DATA_ML")
df = pd.read_csv(df_path)
print(f"✓ 데이터 shape: {df.shape}")

# ==================== 2. 타겟 변수 확인 ====================
target_col = '모형개발용Performance(향후1년내부도여부)'
print(f"\n[2] 타겟 변수: {target_col}")
print(f"\n클래스 분포:")
print(df[target_col].value_counts())
print(f"\n부도 비율: {df[target_col].mean():.2%}")

# ==================== 3. 특성 선택 및 전처리 ====================
print("\n[3] 특성 선택 및 전처리...")

exclude_cols = [
    '기준년월', '업종(중분류)', '설립일자', '주소지시군구',
    target_col, '기업신용평가등급(구간화)'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"초기 특성 수: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[target_col].copy()

# 결측치 처리
missing_ratio = (X.isnull().sum() / len(X) * 100)
high_missing_cols = missing_ratio[missing_ratio > 50].index.tolist()
X = X.drop(columns=high_missing_cols)
print(f"결측치 50% 이상 컬럼 {len(high_missing_cols)}개 제거")

X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"최종 특성 수: {X.shape[1]}")

# ==================== 4. 데이터 분할 ====================
print("\n[4] 데이터 분할 (Train 70% / Test 30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
print(f"\n학습 데이터 클래스 분포:")
print(f"  - 정상 기업: {(y_train==0).sum()}")
print(f"  - 부도 기업: {(y_train==1).sum()}")
print(f"  - 부도율: {y_train.mean():.2%}")

# ==================== 5. SMOTE 적용 ====================
print("\n[5] SMOTE 데이터 증강...")

# SMOTE 적용 전 k_neighbors 설정
n_minority = (y_train==1).sum()
k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1

smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nSMOTE 적용 후 학습 데이터:")
print(f"  - 정상 기업: {(y_train_smote==0).sum()}")
print(f"  - 부도 기업: {(y_train_smote==1).sum()}")
print(f"  - 부도율: {y_train_smote.mean():.2%}")
print(f"  - 총 샘플: {len(y_train_smote)} (증가: {len(y_train_smote) - len(y_train)})")

# ==================== 6. 특성 스케일링 ====================
print("\n[6] 특성 스케일링...")

# 원본 데이터용 스케일러
scaler_original = StandardScaler()
X_train_scaled = scaler_original.fit_transform(X_train)
X_test_scaled = scaler_original.transform(X_test)

# SMOTE 데이터용 스케일러
scaler_smote = StandardScaler()
X_train_smote_scaled = scaler_smote.fit_transform(X_train_smote)
X_test_smote_scaled = scaler_smote.transform(X_test)

print("✓ 스케일링 완료")

# ==================== 7. 모델 정의 ====================
print("\n[7] 모델 정의...")

# 모델 A: 원본 데이터 + class_weight 조정
models_original = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

# 모델 B: SMOTE 증강 데이터
models_smote = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

print("✓ 3가지 알고리즘 준비 완료")

# ==================== 8. 모델 학습 및 평가 ====================
print("\n[8] 모델 학습 및 평가...")
print("=" * 100)

results_original = {}
results_smote = {}

# 8-1. 원본 데이터 모델 학습
print("\n[모델 A] 원본 데이터 + class_weight 조정")
print("-" * 100)

for name, model in models_original.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    results_original[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

# 8-2. SMOTE 데이터 모델 학습
print("\n" + "=" * 100)
print("\n[모델 B] SMOTE 증강 데이터")
print("-" * 100)

for name, model in models_smote.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_smote_scaled, y_train_smote)
    
    y_pred = model.predict(X_test_smote_scaled)
    y_pred_proba = model.predict_proba(X_test_smote_scaled)[:, 1]
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    results_smote[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

# ==================== 9. 성능 비교 테이블 ====================
print("\n" + "=" * 100)
print("[9] 성능 비교 테이블")
print("=" * 100)

comparison_data = []

for name in models_original.keys():
    # 원본 데이터 결과
    comparison_data.append({
        '모델': name,
        '데이터': '원본',
        'Accuracy': results_original[name]['accuracy'],
        'Precision': results_original[name]['precision'],
        'Recall': results_original[name]['recall'],
        'F1 Score': results_original[name]['f1'],
        'ROC AUC': results_original[name]['roc_auc']
    })
    
    # SMOTE 데이터 결과
    comparison_data.append({
        '모델': name,
        '데이터': 'SMOTE',
        'Accuracy': results_smote[name]['accuracy'],
        'Precision': results_smote[name]['precision'],
        'Recall': results_smote[name]['recall'],
        'F1 Score': results_smote[name]['f1'],
        'ROC AUC': results_smote[name]['roc_auc']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n전체 성능 비교:")
print("=" * 120)
print(f"{'모델':<25} {'데이터':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'ROC AUC':<12}")
print("=" * 120)

for _, row in comparison_df.iterrows():
    print(f"{row['모델']:<25} {row['데이터']:<10} {row['Accuracy']:<12.4f} "
          f"{row['Precision']:<12.4f} {row['Recall']:<12.4f} "
          f"{row['F1 Score']:<12.4f} {row['ROC AUC']:<12.4f}")

# ==================== 10. 모델별 성능 향상/하락 분석 ====================
print("\n" + "=" * 100)
print("[10] SMOTE 적용 효과 분석 (원본 대비 변화)")
print("=" * 100)

print(f"\n{'모델':<25} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15} {'ROC AUC':<15}")
print("=" * 100)

for name in models_original.keys():
    acc_diff = results_smote[name]['accuracy'] - results_original[name]['accuracy']
    prec_diff = results_smote[name]['precision'] - results_original[name]['precision']
    rec_diff = results_smote[name]['recall'] - results_original[name]['recall']
    f1_diff = results_smote[name]['f1'] - results_original[name]['f1']
    roc_diff = results_smote[name]['roc_auc'] - results_original[name]['roc_auc']
    
    acc_sign = "📈" if acc_diff > 0 else "📉" if acc_diff < 0 else "➡️"
    prec_sign = "📈" if prec_diff > 0 else "📉" if prec_diff < 0 else "➡️"
    rec_sign = "📈" if rec_diff > 0 else "📉" if rec_diff < 0 else "➡️"
    f1_sign = "📈" if f1_diff > 0 else "📉" if f1_diff < 0 else "➡️"
    roc_sign = "📈" if roc_diff > 0 else "📉" if roc_diff < 0 else "➡️"
    
    print(f"{name:<25} {acc_sign}{acc_diff:+.4f}      {prec_sign}{prec_diff:+.4f}      "
          f"{rec_sign}{rec_diff:+.4f}      {f1_sign}{f1_diff:+.4f}      {roc_sign}{roc_diff:+.4f}")

# ==================== 11. 상세 혼동 행렬 비교 ====================
print("\n" + "=" * 100)
print("[11] 혼동 행렬 비교")
print("=" * 100)

for name in models_original.keys():
    print(f"\n[{name}]")
    print("-" * 60)
    
    cm_orig = results_original[name]['confusion_matrix']
    cm_smote = results_smote[name]['confusion_matrix']
    
    print(f"\n원본 데이터:")
    print(f"  TN: {cm_orig[0,0]:>6}  |  FP: {cm_orig[0,1]:>6}")
    print(f"  FN: {cm_orig[1,0]:>6}  |  TP: {cm_orig[1,1]:>6}")
    
    total_bankrupt = y_test.sum()
    detected_orig = cm_orig[1,1]
    missed_orig = cm_orig[1,0]
    
    print(f"  → 부도 탐지율: {detected_orig}/{total_bankrupt} ({detected_orig/total_bankrupt*100:.1f}%)")
    print(f"  → 부도 놓침:   {missed_orig}/{total_bankrupt} ({missed_orig/total_bankrupt*100:.1f}%)")
    
    print(f"\nSMOTE 데이터:")
    print(f"  TN: {cm_smote[0,0]:>6}  |  FP: {cm_smote[0,1]:>6}")
    print(f"  FN: {cm_smote[1,0]:>6}  |  TP: {cm_smote[1,1]:>6}")
    
    detected_smote = cm_smote[1,1]
    missed_smote = cm_smote[1,0]
    
    print(f"  → 부도 탐지율: {detected_smote}/{total_bankrupt} ({detected_smote/total_bankrupt*100:.1f}%)")
    print(f"  → 부도 놓침:   {missed_smote}/{total_bankrupt} ({missed_smote/total_bankrupt*100:.1f}%)")
    
    # 개선 효과
    improvement = detected_smote - detected_orig
    if improvement > 0:
        print(f"\n  ✅ SMOTE 적용으로 {improvement}개 부도 기업 추가 탐지 ({improvement/total_bankrupt*100:.1f}%p 향상)")
    elif improvement < 0:
        print(f"\n  ⚠️ SMOTE 적용으로 {abs(improvement)}개 부도 기업 탐지 감소")
    else:
        print(f"\n  ➡️ 탐지율 변화 없음")

# ==================== 12. 최고 성능 모델 선정 ====================
print("\n" + "=" * 100)
print("[12] 최고 성능 모델 선정")
print("=" * 100)

# ROC AUC 기준
best_original_name = max(results_original.items(), key=lambda x: x[1]['roc_auc'])[0]
best_original_auc = results_original[best_original_name]['roc_auc']

best_smote_name = max(results_smote.items(), key=lambda x: x[1]['roc_auc'])[0]
best_smote_auc = results_smote[best_smote_name]['roc_auc']

print(f"\n원본 데이터 최고 모델:")
print(f"  {best_original_name} - ROC AUC: {best_original_auc:.4f}")
print(f"  Recall: {results_original[best_original_name]['recall']:.4f}")
print(f"  F1 Score: {results_original[best_original_name]['f1']:.4f}")

print(f"\nSMOTE 데이터 최고 모델:")
print(f"  {best_smote_name} - ROC AUC: {best_smote_auc:.4f}")
print(f"  Recall: {results_smote[best_smote_name]['recall']:.4f}")
print(f"  F1 Score: {results_smote[best_smote_name]['f1']:.4f}")

# 전체 최고 모델
if best_original_auc > best_smote_auc:
    print(f"\n🏆 전체 최고 모델: 원본 데이터 + {best_original_name}")
    print(f"   ROC AUC: {best_original_auc:.4f}")
else:
    print(f"\n🏆 전체 최고 모델: SMOTE 데이터 + {best_smote_name}")
    print(f"   ROC AUC: {best_smote_auc:.4f}")

# ==================== 13. 결과 저장 ====================
print("\n" + "=" * 100)
print("[13] 결과 저장")
print("=" * 100)

# 성능 비교 테이블 저장
comparison_df.to_csv('model_comparison_original_vs_smote.csv', 
                     index=False, encoding='utf-8-sig')
print("✓ 성능 비교 결과 저장: model_comparison_original_vs_smote.csv")

# 모델별 상세 결과 저장
detailed_results = []

for name in models_original.keys():
    cm_orig = results_original[name]['confusion_matrix']
    cm_smote = results_smote[name]['confusion_matrix']
    
    detailed_results.append({
        '모델': name,
        '데이터': '원본',
        'TN': cm_orig[0,0],
        'FP': cm_orig[0,1],
        'FN': cm_orig[1,0],
        'TP': cm_orig[1,1],
        '부도탐지율': f"{cm_orig[1,1]/y_test.sum()*100:.1f}%",
        'Accuracy': results_original[name]['accuracy'],
        'Precision': results_original[name]['precision'],
        'Recall': results_original[name]['recall'],
        'F1 Score': results_original[name]['f1'],
        'ROC AUC': results_original[name]['roc_auc']
    })
    
    detailed_results.append({
        '모델': name,
        '데이터': 'SMOTE',
        'TN': cm_smote[0,0],
        'FP': cm_smote[0,1],
        'FN': cm_smote[1,0],
        'TP': cm_smote[1,1],
        '부도탐지율': f"{cm_smote[1,1]/y_test.sum()*100:.1f}%",
        'Accuracy': results_smote[name]['accuracy'],
        'Precision': results_smote[name]['precision'],
        'Recall': results_smote[name]['recall'],
        'F1 Score': results_smote[name]['f1'],
        'ROC AUC': results_smote[name]['roc_auc']
    })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('detailed_comparison_results.csv', 
                   index=False, encoding='utf-8-sig')
print("✓ 상세 비교 결과 저장: detailed_comparison_results.csv")

# ==================== 14. 최종 요약 ====================
print("\n" + "=" * 100)
print("분석 완료!")
print("=" * 100)

print(f"\n✓ 전체 데이터: {len(df):,}건")
print(f"✓ 사용 특성: {X.shape[1]}개")
print(f"✓ 테스트 데이터: {len(y_test):,}건 (부도: {y_test.sum()}건)")
print(f"\n✓ 비교 모델: 3가지 알고리즘 × 2가지 데이터 = 총 6개 모델")
print(f"  - Logistic Regression")
print(f"  - Random Forest")
print(f"  - Gradient Boosting")
print(f"\n✓ 원본 데이터 학습 샘플: {len(y_train):,}건")
print(f"✓ SMOTE 데이터 학습 샘플: {len(y_train_smote):,}건 (증가: {len(y_train_smote)-len(y_train):,}건)")

print(f"\n생성된 파일:")
print(f"  - model_comparison_original_vs_smote.csv")
print(f"  - detailed_comparison_results.csv")

print("\n" + "=" * 100)