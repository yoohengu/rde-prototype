"""
기업 부도 확률 예측 시스템
- 학습: 불균형 데이터 + class_weight='balanced'
- 예측: 새로운 기업 데이터 입력 → 부도 확률 출력
- Feature Importance 분석 추가
"""
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("기업 부도 확률 예측 시스템")
print("=" * 80)

# ============================================================================
# PART 1: 모델 학습 및 저장
# ============================================================================

def train_and_save_model():
    """모델 학습 및 저장"""
    
    print("\n[PART 1] 모델 학습 및 저장")
    print("=" * 80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    load_dotenv()
    df_path = os.getenv("DATA_ML")
    df = pd.read_csv(df_path)
    print(f"✓ 데이터 shape: {df.shape}")
    
    # 2. 타겟 변수 확인
    target_col = '모형개발용Performance(향후1년내부도여부)'
    print(f"\n[2] 타겟 변수: {target_col}")
    print(f"부도율: {df[target_col].mean():.2%}")
    print(f"부도 기업: {df[target_col].sum()}개 / 전체: {len(df)}개")
    
    # 3. 특성 선택
    print("\n[3] 특성 선택 및 전처리...")
    
    exclude_cols = [
        '기준년월', '업종(중분류)', '설립일자', '주소지시군구',
        target_col, '기업신용평가등급(구간화)'
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"초기 특성 수: {len(feature_cols)}")
    
    # 4. 결측치 처리
    missing_ratio = (X.isnull().sum() / len(X) * 100)
    high_missing_cols = missing_ratio[missing_ratio > 50].index.tolist()
    X = X.drop(columns=high_missing_cols)
    print(f"결측치 50% 이상 컬럼 {len(high_missing_cols)}개 제거")
    
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"최종 특성 수: {X.shape[1]}")
    
    # 최종 feature_cols 저장 (중요!)
    final_feature_cols = X.columns.tolist()
    
    # 5. 데이터 분할
    print("\n[4] 데이터 분할...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"학습 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    
    # 6. 스케일링
    print("\n[5] 특성 스케일링...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. 모델 학습 (Logistic Regression - 최적)
    print("\n[6] Logistic Regression 학습 (class_weight='balanced')...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # 핵심!
        solver='lbfgs'
    )
    
    model.fit(X_train_scaled, y_train)
    print("✓ 학습 완료")
    
    # 8. 모델 평가
    print("\n[7] 모델 평가...")
    
    # 기본 임계값 (0.5)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n임계값 0.5 (기본):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # 최적 임계값 찾기 (Recall 우선)
    print("\n[8] 최적 임계값 탐색...")
    best_threshold = 0.5
    best_f1 = f1
    
    for threshold in np.arange(0.1, 0.6, 0.05):
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        recall_temp = recall_score(y_test, y_pred_temp, zero_division=0)
        precision_temp = precision_score(y_test, y_pred_temp, zero_division=0)
        f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
        
        # Recall이 0.6 이상이면서 F1이 가장 높은 임계값
        if recall_temp >= 0.6 and f1_temp > best_f1:
            best_threshold = threshold
            best_f1 = f1_temp
    
    print(f"최적 임계값: {best_threshold:.2f}")
    
    # 최적 임계값으로 재평가
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    
    accuracy_opt = accuracy_score(y_test, y_pred_optimal)
    precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_opt = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_opt = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    print(f"\n임계값 {best_threshold:.2f} (최적):")
    print(f"  Accuracy:  {accuracy_opt:.4f}")
    print(f"  Precision: {precision_opt:.4f}")
    print(f"  Recall:    {recall_opt:.4f} ⭐")
    print(f"  F1 Score:  {f1_opt:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # 혼동 행렬
    print("\n[9] 혼동 행렬 (최적 임계값):")
    cm = confusion_matrix(y_test, y_pred_optimal)
    print(cm)
    print(f"\nTN: {cm[0,0]} (정상을 정상으로)")
    print(f"FP: {cm[0,1]} (정상을 부도로 - 오탐)")
    print(f"FN: {cm[1,0]} (부도를 정상으로 - 놓침) ⚠️")
    print(f"TP: {cm[1,1]} (부도를 부도로 - 정답) ✓")
    
    # 실제 의미
    total_bankrupt = y_test.sum()
    detected = cm[1,1]
    missed = cm[1,0]
    
    print(f"\n실제 부도 기업: {total_bankrupt}개")
    print(f"✓ 탐지: {detected}개 ({detected/total_bankrupt*100:.1f}%)")
    print(f"✗ 놓침: {missed}개 ({missed/total_bankrupt*100:.1f}%)")
    
    # ⭐⭐⭐ 10. Feature Importance 분석 (Logistic Regression) ⭐⭐⭐
    print("\n[10] Feature Importance 분석...")
    
    # Logistic Regression의 계수(coefficient) 추출
    coefficients = model.coef_[0]  # shape: (n_features,)
    
    # 절댓값 기준으로 중요도 계산 (양수/음수 모두 중요)
    feature_importance_df = pd.DataFrame({
        'feature': final_feature_cols,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'importance_rank': None
    })
    
    # 절댓값 기준으로 정렬
    feature_importance_df = feature_importance_df.sort_values(
        'abs_coefficient', 
        ascending=False
    ).reset_index(drop=True)
    
    # 순위 매기기
    feature_importance_df['importance_rank'] = range(1, len(feature_importance_df) + 1)
    
    # 영향 방향 추가 (부도 위험 증가 vs 감소)
    feature_importance_df['impact'] = feature_importance_df['coefficient'].apply(
        lambda x: '부도 위험 증가 ⬆️' if x > 0 else '부도 위험 감소 ⬇️'
    )
    
    # 중요도 비율 (%) 계산
    total_abs_coef = feature_importance_df['abs_coefficient'].sum()
    feature_importance_df['importance_percent'] = (
        feature_importance_df['abs_coefficient'] / total_abs_coef * 100
    ).round(2)
    
    # Top 20 출력
    print("\n특성 중요도 Top 20:")
    print("=" * 100)
    print(f"{'순위':<5} {'특성명':<40} {'계수':<15} {'절댓값':<15} {'영향':<20} {'중요도(%)'}")
    print("=" * 100)
    
    for idx, row in feature_importance_df.head(20).iterrows():
        print(f"{row['importance_rank']:<5} {row['feature']:<40} "
              f"{row['coefficient']:<15.6f} {row['abs_coefficient']:<15.6f} "
              f"{row['impact']:<20} {row['importance_percent']:.2f}%")
    
    # 전체 Feature Importance CSV 저장
    feature_importance_df.to_csv(
        'feature_importance_logistic.csv', 
        index=False, 
        encoding='utf-8-sig'
    )
    print(f"\n✓ 전체 Feature Importance 저장: feature_importance_logistic.csv")
    
    # Top 50만 별도 저장
    feature_importance_df.head(50).to_csv(
        'feature_importance_top50.csv', 
        index=False, 
        encoding='utf-8-sig'
    )
    print(f"✓ Top 50 Feature Importance 저장: feature_importance_top50.csv")
    
    # 부도 위험 증가 요인 Top 10
    increasing_risk = feature_importance_df[
        feature_importance_df['coefficient'] > 0
    ].head(10)
    
    increasing_risk.to_csv(
        'feature_importance_risk_increasing.csv', 
        index=False, 
        encoding='utf-8-sig'
    )
    print(f"✓ 부도 위험 증가 요인 Top 10 저장: feature_importance_risk_increasing.csv")
    
    # 부도 위험 감소 요인 Top 10
    decreasing_risk = feature_importance_df[
        feature_importance_df['coefficient'] < 0
    ].head(10)
    
    decreasing_risk.to_csv(
        'feature_importance_risk_decreasing.csv', 
        index=False, 
        encoding='utf-8-sig'
    )
    print(f"✓ 부도 위험 감소 요인 Top 10 저장: feature_importance_risk_decreasing.csv")
    
    # 11. 모델 저장
    print("\n[11] 모델 저장 중...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 저장할 객체들
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_cols': final_feature_cols,
        'optimal_threshold': best_threshold,
        'training_date': timestamp,
        'metrics': {
            'roc_auc': roc_auc,
            'recall': recall_opt,
            'precision': precision_opt,
            'f1_score': f1_opt,
            'accuracy': accuracy_opt
        },
        'feature_importance': feature_importance_df  # ⭐ Feature Importance 추가
    }
    
    # joblib로 저장 (pickle보다 효율적)
    model_filename = f'bankruptcy_model_{timestamp}.pkl'
    joblib.dump(model_package, model_filename)
    
    print(f"✓ 모델 저장 완료: {model_filename}")
    
    # 최신 모델로도 저장 (로드하기 편하게)
    joblib.dump(model_package, 'bankruptcy_model_latest.pkl')
    print(f"✓ 최신 모델 저장: bankruptcy_model_latest.pkl")
    
    # 메타데이터 저장
    metadata = pd.DataFrame([{
        'timestamp': timestamp,
        'n_samples': len(df),
        'n_features': len(final_feature_cols),
        'default_rate': y.mean(),
        'roc_auc': roc_auc,
        'recall': recall_opt,
        'precision': precision_opt,
        'f1_score': f1_opt,
        'optimal_threshold': best_threshold
    }])
    
    metadata.to_csv('model_metadata.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 메타데이터 저장: model_metadata.csv")
    
    print("\n" + "=" * 80)
    print("학습 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  📊 모델 파일:")
    print(f"    - {model_filename}")
    print(f"    - bankruptcy_model_latest.pkl")
    print("  📈 Feature Importance:")
    print("    - feature_importance_logistic.csv (전체)")
    print("    - feature_importance_top50.csv (상위 50개)")
    print("    - feature_importance_risk_increasing.csv (부도 위험 증가 요인)")
    print("    - feature_importance_risk_decreasing.csv (부도 위험 감소 요인)")
    print("  📋 메타데이터:")
    print("    - model_metadata.csv")
    
    return model_package


# ============================================================================
# PART 2: 부도 확률 예측 함수
# ============================================================================

def predict_bankruptcy_probability(company_data, model_path='bankruptcy_model_latest.pkl'):
    """
    새로운 기업 데이터로 부도 확률 예측
    
    Parameters:
    -----------
    company_data : dict or pd.DataFrame
        기업 재무 데이터 (딕셔너리 또는 DataFrame)
    model_path : str
        저장된 모델 파일 경로
    
    Returns:
    --------
    result : dict
        {
            'probability': 부도 확률 (0~1),
            'risk_level': 위험 등급,
            'prediction': 부도 예측 (0 or 1),
            'confidence': 예측 신뢰도
        }
    """
    
    # 1. 모델 로드
    model_package = joblib.load(model_path)
    model = model_package['model']
    scaler = model_package['scaler']
    feature_cols = model_package['feature_cols']
    optimal_threshold = model_package['optimal_threshold']
    
    # 2. 입력 데이터 처리
    if isinstance(company_data, dict):
        df_input = pd.DataFrame([company_data])
    else:
        df_input = company_data.copy()
    
    # 3. 필요한 특성만 추출
    missing_features = set(feature_cols) - set(df_input.columns)
    if missing_features:
        print(f"경고: 누락된 특성 {len(missing_features)}개를 0으로 채웁니다.")
        for col in missing_features:
            df_input[col] = 0
    
    X_input = df_input[feature_cols].copy()
    
    # 4. 전처리 (결측치, 무한대)
    X_input = X_input.fillna(0)
    X_input = X_input.replace([np.inf, -np.inf], 0)
    
    # 5. 스케일링
    X_input_scaled = scaler.transform(X_input)
    
    # 6. 예측
    probability = model.predict_proba(X_input_scaled)[:, 1][0]
    prediction = 1 if probability >= optimal_threshold else 0
    
    # 7. 위험 등급 분류
    if probability < 0.2:
        risk_level = "매우 낮음"
        risk_color = "🟢"
    elif probability < 0.4:
        risk_level = "낮음"
        risk_color = "🟡"
    elif probability < 0.6:
        risk_level = "보통"
        risk_color = "🟠"
    elif probability < 0.8:
        risk_level = "높음"
        risk_color = "🔴"
    else:
        risk_level = "매우 높음"
        risk_color = "🚨"
    
    # 8. 신뢰도 계산 (확률이 0 또는 1에 가까울수록 높음)
    confidence = abs(probability - 0.5) * 2  # 0~1 범위로 정규화
    
    # 9. 결과 반환
    result = {
        'probability': float(probability),
        'probability_percent': f"{probability * 100:.2f}%",
        'prediction': int(prediction),
        'prediction_label': "부도 예상" if prediction == 1 else "정상",
        'risk_level': risk_level,
        'risk_color': risk_color,
        'confidence': float(confidence),
        'threshold': optimal_threshold
    }
    
    return result


# ============================================================================
# PART 3: 배치 예측 (여러 기업 동시 예측)
# ============================================================================

def predict_batch(df_companies, model_path='bankruptcy_model_latest.pkl'):
    """
    여러 기업 동시 예측
    
    Parameters:
    -----------
    df_companies : pd.DataFrame
        여러 기업의 재무 데이터
    model_path : str
        저장된 모델 파일 경로
    
    Returns:
    --------
    df_results : pd.DataFrame
        예측 결과가 추가된 DataFrame
    """
    
    # 모델 로드
    model_package = joblib.load(model_path)
    model = model_package['model']
    scaler = model_package['scaler']
    feature_cols = model_package['feature_cols']
    optimal_threshold = model_package['optimal_threshold']
    
    # 특성 추출
    missing_features = set(feature_cols) - set(df_companies.columns)
    if missing_features:
        print(f"경고: 누락된 특성 {len(missing_features)}개를 0으로 채웁니다.")
        for col in missing_features:
            df_companies[col] = 0
    
    X = df_companies[feature_cols].copy()
    
    # 전처리
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # 스케일링
    X_scaled = scaler.transform(X)
    
    # 예측
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= optimal_threshold).astype(int)
    
    # 위험 등급
    risk_levels = []
    for prob in probabilities:
        if prob < 0.2:
            risk_levels.append("매우 낮음")
        elif prob < 0.4:
            risk_levels.append("낮음")
        elif prob < 0.6:
            risk_levels.append("보통")
        elif prob < 0.8:
            risk_levels.append("높음")
        else:
            risk_levels.append("매우 높음")
    
    # 결과 추가
    df_results = df_companies.copy()
    df_results['부도확률'] = probabilities
    df_results['부도확률(%)'] = (probabilities * 100).round(2)
    df_results['부도예측'] = predictions
    df_results['위험등급'] = risk_levels
    
    return df_results


# ============================================================================
# PART 4: 사용 예시
# ============================================================================

if __name__ == "__main__":
    
    # ========================================
    # 1. 모델 학습 및 저장 (최초 1회만)
    # ========================================
    
    print("\n" + "=" * 80)
    print("모델을 학습하시겠습니까? (y/n)")
    print("(최초 실행 또는 모델 업데이트 시 'y' 입력)")
    print("=" * 80)
    
    choice = input("입력: ").strip().lower()
    
    if choice == 'y':
        model_package = train_and_save_model()
        print("\n✓ 모델 학습 및 저장 완료!")
    
    # ========================================
    # 2. CSV 파일에서 데이터 읽어서 예측 ⭐ 새로 추가!
    # ========================================
    
    print("\n" + "=" * 80)
    print("[PART 2] CSV 파일에서 데이터 읽어서 예측")
    print("=" * 80)
    
    csv_file = r'C:\Users\user\rde-data\test.csv'  # 또는 사용자 입력받기
    
    try:
        # CSV 파일 존재 확인
        if not os.path.exists(csv_file):
            print(f"\n⚠️  '{csv_file}' 파일을 찾을 수 없습니다.")
            print("   파일 경로를 확인해주세요.")
        else:
            print(f"\n✓ '{csv_file}' 파일을 읽는 중...")
            
            # CSV 파일 읽기 (탭 구분자 시도)
            try:
                df_test = pd.read_csv(csv_file, sep='\t')
                print(f"✓ 탭 구분자로 읽기 성공")
            except:
                # 쉼표 구분자 시도
                df_test = pd.read_csv(csv_file)
                print(f"✓ 쉼표 구분자로 읽기 성공")
            
            print(f"✓ 데이터 로드 완료: {len(df_test)}개 기업")
            print(f"✓ 컬럼 수: {len(df_test.columns)}")
            
            # 예측 실행
            print("\n예측 진행 중...")
            result = predict_bankruptcy_probability(df_test.iloc[0].to_dict())
            
            # 결과 출력
            print("\n" + "-" * 80)
            print("예측 결과:")
            print("-" * 80)
            print(f"기업명/ID:     (CSV 1번째 행)")
            print(f"부도 확률:     {result['probability_percent']} {result['risk_color']}")
            print(f"위험 등급:     {result['risk_level']}")
            print(f"예측 결과:     {result['prediction_label']}")
            print(f"예측 신뢰도:   {result['confidence']:.2%}")
            print(f"사용 임계값:   {result['threshold']:.2f}")
            print("-" * 80)
            
            if result['prediction'] == 1:
                print("\n⚠️  이 기업은 향후 1년 내 부도 위험이 높습니다!")
                print("    추가 심사 또는 대출 거절을 권장합니다.")
            else:
                print("\n✓  이 기업은 재무적으로 안정적입니다.")
                print("    대출 승인 가능 (단, 추가 검토 권장)")
            
            # 여러 기업 예측 옵션
            if len(df_test) > 1:
                print("\n" + "-" * 80)
                print(f"CSV 파일에 {len(df_test)}개 기업이 있습니다.")
                print("모든 기업을 예측하시겠습니까? (y/n)")
                print("-" * 80)
                
                batch_choice = input("입력: ").strip().lower()
                
                if batch_choice == 'y':
                    print("\n배치 예측 진행 중...")
                    df_results = predict_batch(df_test)
                    
                    # 결과 저장
                    output_file = 'predictions_result.csv'
                    df_results[['부도확률(%)', '부도예측', '위험등급']].to_csv(
                        output_file, 
                        index=False, 
                        encoding='utf-8-sig'
                    )
                    
                    print(f"\n✓ 예측 완료: {len(df_results)}개 기업")
                    print(f"✓ 결과 저장: {output_file}")
                    
                    # 요약 통계
                    print("\n" + "-" * 80)
                    print("예측 요약:")
                    print("-" * 80)
                    print(f"전체 기업 수:     {len(df_results)}")
                    print(f"부도 예상:        {df_results['부도예측'].sum()}개 ({df_results['부도예측'].sum()/len(df_results)*100:.1f}%)")
                    print(f"정상 예상:        {(df_results['부도예측']==0).sum()}개 ({(df_results['부도예측']==0).sum()/len(df_results)*100:.1f}%)")
                    print(f"\n평균 부도 확률:   {df_results['부도확률(%)'].mean():.2f}%")
                    print(f"최고 부도 확률:   {df_results['부도확률(%)'].max():.2f}%")
                    print(f"최저 부도 확률:   {df_results['부도확률(%)'].min():.2f}%")
                    print("-" * 80)
                    
                    # 위험 등급별 분포
                    print("\n위험 등급 분포:")
                    print(df_results['위험등급'].value_counts().sort_index())
    
    except FileNotFoundError:
        print("\n❌ 저장된 모델이 없습니다.")
        print("   먼저 모델을 학습해주세요. (프로그램 시작 시 'y' 입력)")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # 4. 프로그램 종료
    # ========================================
    
    print("\n" + "=" * 80)
    print("프로그램 종료")
    print("=" * 80)