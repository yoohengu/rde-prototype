"""
기업 부도 예측 머신러닝 모델 (개선 버전)
- 목표변수: 모형개발용Performance(향후1년내부도여부) 
- 다양한 샘플링 기법 + 임계값 조정 + 가중치 조정
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
# 다양한 샘플링 기법
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("기업 부도 예측 머신러닝 모델 (개선 버전)")
print("=" * 80)

# ==================== 설정 옵션 ====================
# 여기서 원하는 방법을 선택하세요!

SAMPLING_METHOD = 'BorderlineSMOTE'  # 옵션: 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'SMOTETomek', 'UnderSampling', 'None'
SAMPLING_RATIO = 0.5  # 0.5 = 부도:정상 = 1:2, 1.0 = 완전 균형
USE_THRESHOLD_TUNING = True  # 임계값 조정 사용 여부
CLASS_WEIGHT_MULTIPLIER = 10  # class_weight에 적용할 가중치 (1~20 권장)

print(f"\n[설정]")
print(f"  - 샘플링 방법: {SAMPLING_METHOD}")
print(f"  - 샘플링 비율: {SAMPLING_RATIO}")
print(f"  - 임계값 조정: {USE_THRESHOLD_TUNING}")
print(f"  - 부도 가중치: x{CLASS_WEIGHT_MULTIPLIER}")
# ================================================

# 1. 데이터 로드
print("\n[1] 데이터 로드 중...")
load_dotenv()
df_path = os.getenv("DATA_ML")
df = pd.read_csv(df_path)
print(f"데이터 shape: {df.shape}")
print(f"컬럼 수: {len(df.columns)}")

# 2. 타겟 변수 확인
target_col = '모형개발용Performance(향후1년내부도여부)'
print(f"\n[2] 타겟 변수: {target_col}")
print(f"\n부도 비율:")
print(df[target_col].value_counts(normalize=True))
print(f"\n클래스 분포:")
print(df[target_col].value_counts())

# 3. 특성 선택 및 전처리
print("\n[3] 특성 선택 및 전처리...")

exclude_cols = [
    '기준년월', '업종(중분류)', '설립일자', '주소지시군구',
    target_col, '기업신용평가등급(구간화)'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"사용할 특성 수: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[target_col].copy()

# 결측치 처리
print(f"\n결측치 비율 (상위 10개):")
missing_ratio = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
print(missing_ratio.head(10))

high_missing_cols = missing_ratio[missing_ratio > 50].index.tolist()
X = X.drop(columns=high_missing_cols)
print(f"\n결측치 50% 이상 컬럼 {len(high_missing_cols)}개 제거")

X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"전처리 후 특성 수: {X.shape[1]}")

# 4. 데이터 분할
print("\n[4] 데이터 분할 (Train 70% / Test 30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
print(f"학습 데이터 부도율: {y_train.mean():.2%}")
print(f"테스트 데이터 부도율: {y_test.mean():.2%}")

# 5. 샘플링 적용
print(f"\n[5] 샘플링 적용: {SAMPLING_METHOD}...")
print(f"샘플링 전:")
print(f"  - 정상 기업: {(y_train==0).sum()}")
print(f"  - 부도 기업: {(y_train==1).sum()}")
print(f"  - 총 샘플: {len(y_train)}")

n_minority = (y_train==1).sum()
k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1

if SAMPLING_METHOD == 'SMOTE':
    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=SAMPLING_RATIO)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
elif SAMPLING_METHOD == 'BorderlineSMOTE':
    sampler = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=SAMPLING_RATIO)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
elif SAMPLING_METHOD == 'ADASYN':
    sampler = ADASYN(random_state=42, n_neighbors=k_neighbors, sampling_strategy=SAMPLING_RATIO)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
elif SAMPLING_METHOD == 'SMOTETomek':
    sampler = SMOTETomek(random_state=42, sampling_strategy=SAMPLING_RATIO)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
elif SAMPLING_METHOD == 'UnderSampling':
    sampler = RandomUnderSampler(random_state=42, sampling_strategy=SAMPLING_RATIO)
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
else:  # 'None'
    X_train_resampled, y_train_resampled = X_train, y_train
    print("  (샘플링 적용 안 함)")

if SAMPLING_METHOD != 'None':
    print(f"\n샘플링 후:")
    print(f"  - 정상 기업: {(y_train_resampled==0).sum()}")
    print(f"  - 부도 기업: {(y_train_resampled==1).sum()}")
    print(f"  - 총 샘플: {len(y_train_resampled)}")

# 6. 특성 스케일링
print("\n[6] 특성 스케일링...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 7. 모델 학습
print("\n[7] 모델 학습 중...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight={0: 1, 1: CLASS_WEIGHT_MULTIPLIER}  # 가중치 조정
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        class_weight={0: 1, 1: CLASS_WEIGHT_MULTIPLIER},  # 가중치 조정
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_scaled, y_train_resampled)
    
    # 예측 (기본 임계값 0.5)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 평가
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

# 8. 임계값 조정 (선택사항)
if USE_THRESHOLD_TUNING:
    print("\n[8] 임계값 조정 분석...")
    print("\n모델별 최적 임계값 찾기:")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("="*50)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred_adjusted = (result['y_pred_proba'] >= threshold).astype(int)
            precision = precision_score(y_test, y_pred_adjusted, zero_division=0)
            recall = recall_score(y_test, y_pred_adjusted, zero_division=0)
            f1 = f1_score(y_test, y_pred_adjusted, zero_division=0)
            
            print(f"{threshold:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"→ 최적 임계값: {best_threshold} (F1: {best_f1:.4f})")
        
        # 최적 임계값으로 재예측
        y_pred_best = (result['y_pred_proba'] >= best_threshold).astype(int)
        results[name]['y_pred_best'] = y_pred_best
        results[name]['best_threshold'] = best_threshold
        results[name]['accuracy_best'] = accuracy_score(y_test, y_pred_best)
        results[name]['precision_best'] = precision_score(y_test, y_pred_best, zero_division=0)
        results[name]['recall_best'] = recall_score(y_test, y_pred_best, zero_division=0)
        results[name]['f1_best'] = f1_score(y_test, y_pred_best, zero_division=0)

# 9. 최종 성능 비교
print("\n[9] 최종 모델 성능 비교...")

if USE_THRESHOLD_TUNING:
    print("\n임계값 조정 후 성능:")
    print("="*95)
    print(f"{'Model':<25} {'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("="*95)
    
    for name, result in results.items():
        print(f"{name:<25} {result['best_threshold']:<12.1f} {result['accuracy_best']:<12.4f} "
              f"{result['precision_best']:<12.4f} {result['recall_best']:<12.4f} {result['f1_best']:<12.4f}")
else:
    print("\n기본 임계값(0.5) 성능:")
    print("="*85)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12}")
    print("="*85)
    
    for name, result in results.items():
        print(f"{name:<25} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} {result['f1']:<12.4f} {result['roc_auc']:<12.4f}")

# ROC AUC 기준으로 최고 모델 선택
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_result = results[best_model_name]
best_model = best_result['model']

print(f"\n최고 성능 모델: {best_model_name} (ROC AUC: {best_result['roc_auc']:.4f})")

# 10. 상세 평가
print("\n[10] 최고 모델 상세 평가...")

if USE_THRESHOLD_TUNING:
    y_pred_final = best_result['y_pred_best']
    print(f"(최적 임계값 {best_result['best_threshold']} 적용)")
else:
    y_pred_final = best_result['y_pred']

print("\n분류 리포트:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['정상', '부도'],
                          zero_division=0))

print("\n혼동 행렬:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)
print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# 11. 특성 중요도 분석
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n[11] 특성 중요도 (Top 20)...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20))

# 12. 결과 저장
print("\n[12] 결과 요약 저장 중...")
summary = {
    '모델명': [],
    'Accuracy': [],
    'Precision (부도)': [],
    'Recall (부도)': [],
    'F1 Score': [],
    'ROC AUC': []
}

for name, result in results.items():
    summary['모델명'].append(name)
    if USE_THRESHOLD_TUNING:
        summary['Accuracy'].append(result['accuracy_best'])
        summary['Precision (부도)'].append(result['precision_best'])
        summary['Recall (부도)'].append(result['recall_best'])
        summary['F1 Score'].append(result['f1_best'])
    else:
        summary['Accuracy'].append(result['accuracy'])
        summary['Precision (부도)'].append(result['precision'])
        summary['Recall (부도)'].append(result['recall'])
        summary['F1 Score'].append(result['f1'])
    summary['ROC AUC'].append(result['roc_auc'])

summary_df = pd.DataFrame(summary)
summary_df.to_csv('model_comparison_results_improved.csv', index=False, encoding='utf-8-sig')
print("✓ 모델 비교 결과 저장: model_comparison_results_improved.csv")

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance.to_csv('feature_importance_results_improved.csv',
                             index=False, encoding='utf-8-sig')
    print("✓ 특성 중요도 저장: feature_importance_results_improved.csv")

# 13. 최종 요약
print("\n" + "="*80)
print("분석 완료! (개선 버전)")
print("="*80)
print(f"\n✓ 데이터 샘플 수: {len(df)}")
print(f"✓ 사용된 특성 수: {X.shape[1]}")
print(f"✓ 부도 기업 비율: {y.mean():.2%}")
print(f"\n✓ 샘플링 방법: {SAMPLING_METHOD}")
print(f"✓ 샘플링 전 학습 데이터: {len(y_train)} (부도: {(y_train==1).sum()})")
print(f"✓ 샘플링 후 학습 데이터: {len(y_train_resampled)} (부도: {(y_train_resampled==1).sum()})")
print(f"✓ 부도 가중치: x{CLASS_WEIGHT_MULTIPLIER}")
print(f"\n✓ 최고 성능 모델: {best_model_name}")
print(f"✓ 테스트 세트 ROC AUC: {best_result['roc_auc']:.4f}")

if USE_THRESHOLD_TUNING:
    print(f"✓ 최적 임계값: {best_result['best_threshold']}")
    print(f"✓ 조정 후 F1 Score: {best_result['f1_best']:.4f}")
    print(f"✓ 조정 후 Recall: {best_result['recall_best']:.4f}")
else:
    print(f"✓ 테스트 세트 F1 Score: {best_result['f1']:.4f}")

print("\n생성된 파일:")
print("  - model_comparison_results_improved.csv")
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("  - feature_importance_results_improved.csv")

print("\n" + "="*80)