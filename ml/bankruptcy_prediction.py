"""
기업 부도 예측 머신러닝 모델 (VSCode용)
- 목표변수: 모형개발용Performance(향후1년내부도여부) 
- 다양한 재무비율과 신용정보를 활용한 예측 모델
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
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# Mac의 경우: plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("기업 부도 예측 머신러닝 모델")
print("=" * 80)

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

# 제외할 컬럼들
exclude_cols = [
    '기준년월', '업종(중분류)', '설립일자', '주소지시군구',
    target_col, '기업신용평가등급(구간화)'
]

# 수치형 컬럼만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"사용할 특성 수: {len(feature_cols)}")

# 특성과 타겟 분리
X = df[feature_cols].copy()
y = df[target_col].copy()

# 결측치 확인
print(f"\n결측치 비율 (상위 10개):")
missing_ratio = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
print(missing_ratio.head(10))

# 결측치가 50% 이상인 컬럼 제거
high_missing_cols = missing_ratio[missing_ratio > 50].index.tolist()
X = X.drop(columns=high_missing_cols)
print(f"\n결측치 50% 이상 컬럼 {len(high_missing_cols)}개 제거")

# 나머지 결측치는 중앙값으로 대체
X = X.fillna(X.median())

# 무한대 값 처리
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

# 5. 특성 스케일링
print("\n[5] 특성 스케일링...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 모델 학습
print("\n[6] 모델 학습 중...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
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

results = {}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 평가
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

# 7. 최고 성능 모델 선택
print("\n[7] 모델 성능 비교...")
print("\n" + "="*70)
print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'ROC AUC':<12}")
print("="*70)

for name, result in results.items():
    print(f"{name:<25} {result['accuracy']:<12.4f} {result['f1']:<12.4f} {result['roc_auc']:<12.4f}")

# ROC AUC 기준으로 최고 모델 선택
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_result = results[best_model_name]
best_model = best_result['model']

print(f"\n최고 성능 모델: {best_model_name} (ROC AUC: {best_result['roc_auc']:.4f})")

# 8. 상세 평가
print("\n[8] 최고 모델 상세 평가...")
print("\n분류 리포트:")
print(classification_report(y_test, best_result['y_pred'], 
                          target_names=['정상', '부도'],
                          zero_division=0))

print("\n혼동 행렬:")
cm = confusion_matrix(y_test, best_result['y_pred'])
print(cm)
print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# 9. 특성 중요도 분석 (Random Forest 또는 Gradient Boosting인 경우)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n[9] 특성 중요도 (Top 20)...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20))

# 10. 결과 요약 저장
print("\n[10] 결과 요약 저장 중...")
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
    summary['Accuracy'].append(result['accuracy'])
    summary['Precision (부도)'].append(precision_score(y_test, result['y_pred'], zero_division=0))
    summary['Recall (부도)'].append(recall_score(y_test, result['y_pred'], zero_division=0))
    summary['F1 Score'].append(result['f1'])
    summary['ROC AUC'].append(result['roc_auc'])

summary_df = pd.DataFrame(summary)
summary_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
print("✓ 모델 비교 결과 저장: model_comparison_results.csv")

# 특성 중요도 저장 (Random Forest 또는 Gradient Boosting인 경우)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance.to_csv('feature_importance_results.csv', 
                             index=False, encoding='utf-8-sig')
    print("✓ 특성 중요도 저장: feature_importance_results.csv")

# 11. 최종 요약
print("\n" + "="*80)
print("분석 완료!")
print("="*80)
print(f"\n✓ 데이터 샘플 수: {len(df)}")
print(f"✓ 사용된 특성 수: {X.shape[1]}")
print(f"✓ 부도 기업 비율: {y.mean():.2%}")
print(f"✓ 최고 성능 모델: {best_model_name}")
print(f"✓ 테스트 세트 ROC AUC: {best_result['roc_auc']:.4f}")
print(f"✓ 테스트 세트 F1 Score: {best_result['f1']:.4f}")

print("\n생성된 파일:")
print("  - model_comparison_results.csv (모델 비교 결과)")
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("  - feature_importance_results.csv (특성 중요도)")

print("\n" + "="*80)