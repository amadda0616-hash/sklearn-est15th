# ============================================================
# Seoul Bike 예측 모델 개선 코드
# 복사하여 노트북에 붙여넣기 하세요
# ============================================================

"""
셀 수정 가이드:
- Cell 77: 결측치 처리 함수 → 아래 코드로 교체
- Cell 78: 이상치 처리 → 아래 코드로 교체 (이상치 클리핑 추가)
- Cell 79: 특성 엔지니어링 → 아래 코드로 교체 (대기질 특성 추가)
- Cell 82: 학습/검증 데이터 분리 → 아래 코드로 교체 (시계열 분할 + 로그 변환)
- Cell 84: 모델 학습 → 아래 코드로 교체 (스태킹 제어 + Early Stopping)
- Cell 86+ (신규): Feature Importance 분석 추가
- Cell 87+ (신규): 예측 및 역변환 추가
"""

# ============================================================
# [Cell 77] 결측치 처리 함수 - 개선 버전
# ============================================================
# 기존 코드를 아래로 교체하세요

def fill_missing_advanced(df):
    """
    개선된 결측치 처리 함수
    - 기본 날씨 변수: 시간별 평균
    - 대기질 변수 (오존, PM10, PM2.5): 시간+온도 그룹별 중앙값
    """
    df_filled = df.copy()
    
    # 기본 날씨 변수: 시간별 평균으로 대체
    basic_cols = ['hour_bef_temperature', 'hour_bef_precipitation', 
                  'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility']
    
    for col in basic_cols:
        if col in df_filled.columns and df_filled[col].isnull().sum() > 0:
            hour_mean = df_filled.groupby('hour')[col].transform('mean')
            df_filled[col] = df_filled[col].fillna(hour_mean)
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    # 온도 구간 생성 (대기질 결측치 처리용)
    df_filled['temp_bin'] = pd.cut(
        df_filled['hour_bef_temperature'], 
        bins=5, 
        labels=False
    )
    df_filled['temp_bin'] = df_filled['temp_bin'].fillna(0).astype(int)
    
    # 대기질 변수: 시간+온도 그룹별 중앙값으로 대체
    air_quality_cols = ['hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
    
    for col in air_quality_cols:
        if col in df_filled.columns and df_filled[col].isnull().sum() > 0:
            # 시간+온도 그룹별 중앙값
            group_median = df_filled.groupby(['hour', 'temp_bin'])[col].transform('median')
            df_filled[col] = df_filled[col].fillna(group_median)
            # 여전히 결측치가 있다면 시간별 중앙값
            hour_median = df_filled.groupby('hour')[col].transform('median')
            df_filled[col] = df_filled[col].fillna(hour_median)
            # 최종적으로 전체 중앙값
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    
    # 임시 컬럼 제거
    df_filled.drop('temp_bin', axis=1, inplace=True)
    
    return df_filled

# 적용
train_df = fill_missing_advanced(train_df)
test_df = fill_missing_advanced(test_df)

print('결측치 처리 후:')
print(f'Train 결측치: {train_df.isnull().sum().sum()}')
print(f'Test 결측치: {test_df.isnull().sum().sum()}')


# ============================================================
# [Cell 78] 이상치 처리 - 개선 버전 (클리핑 추가)
# ============================================================
# 기존 코드를 아래로 교체하세요

def detect_outliers_iqr(df, column):
    """IQR 방식 이상치 감지"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def handle_outliers(df, columns):
    """이상치를 경계값으로 클리핑"""
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = max(0, Q1 - 1.5 * IQR)  # 음수 방지 (풍속 등)
            upper = Q3 + 1.5 * IQR
            
            # 이상치를 경계값으로 클리핑
            original_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            print(f'{col}: {original_outliers}개 이상치 클리핑 처리 (범위: {lower:.2f} ~ {upper:.2f})')
    
    return df_clean

# 이상치 분석 및 처리
print('=== 이상치 분석 및 처리 ===')
outlier_cols = ['hour_bef_windspeed', 'hour_bef_pm10', 'hour_bef_pm2.5']
train_df = handle_outliers(train_df, outlier_cols)
test_df = handle_outliers(test_df, outlier_cols)


# ============================================================
# [Cell 79] 특성 엔지니어링 - 개선 버전 (대기질 특성 추가)
# ============================================================
# 기존 코드를 아래로 교체하세요

def create_features(df):
    """
    개선된 특성 엔지니어링 함수
    - 시간 관련 특성
    - 기상 관련 특성
    - 대기질 특성 (신규 추가)
    - 상호작용 특성
    """
    df_fe = df.copy()
    
    # ========== 시간 관련 특성 ==========
    # 출퇴근 시간 여부 (7-9시, 18-20시)
    df_fe['is_rush_hour'] = ((df_fe['hour'] >= 7) & (df_fe['hour'] <= 9) | 
                            (df_fe['hour'] >= 18) & (df_fe['hour'] <= 20)).astype(int)
    
    # 낮/밤 구분 (6시-18시: 낮)
    df_fe['is_daytime'] = ((df_fe['hour'] >= 6) & (df_fe['hour'] < 18)).astype(int)
    
    # 시간대 구분 (새벽/아침/점심/저녁/밤)
    def get_time_period(hour):
        if hour < 6:
            return 0  # 새벽
        elif hour < 12:
            return 1  # 아침
        elif hour < 14:
            return 2  # 점심
        elif hour < 18:
            return 3  # 오후
        elif hour < 22:
            return 4  # 저녁
        else:
            return 5  # 밤
    
    df_fe['time_period'] = df_fe['hour'].apply(get_time_period)
    
    # 시간의 주기적 특성 (sin/cos 변환)
    df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
    
    # ========== 기상 관련 특성 ==========
    # 체감온도 근사 (온도, 습도, 풍속 고려)
    df_fe['feels_like'] = df_fe['hour_bef_temperature'] - \
                         0.55 * (1 - df_fe['hour_bef_humidity']/100) * \
                         (df_fe['hour_bef_temperature'] - 14.5)
    
    # 불쾌지수
    df_fe['discomfort_index'] = 0.81 * df_fe['hour_bef_temperature'] + \
                                0.01 * df_fe['hour_bef_humidity'] * \
                                (0.99 * df_fe['hour_bef_temperature'] - 14.3) + 46.3
    
    # 날씨 좋음 지표 (비 안오고, 적당한 온도, 낮은 미세먼지)
    df_fe['good_weather'] = ((df_fe['hour_bef_precipitation'] == 0) & 
                            (df_fe['hour_bef_temperature'] >= 10) & 
                            (df_fe['hour_bef_temperature'] <= 25) &
                            (df_fe['hour_bef_pm10'] < 80)).astype(int)
    
    # ========== 대기질 특성 (신규 추가) ==========
    # PM10 기준 대기질 등급 (WHO 기준)
    df_fe['pm10_grade'] = pd.cut(
        df_fe['hour_bef_pm10'],
        bins=[-np.inf, 30, 50, 100, np.inf],
        labels=[0, 1, 2, 3]  # 좋음, 보통, 나쁨, 매우나쁨
    ).astype(int)
    
    # PM2.5 기준 대기질 등급
    df_fe['pm25_grade'] = pd.cut(
        df_fe['hour_bef_pm2.5'],
        bins=[-np.inf, 15, 35, 75, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # 오존 위험도 등급
    df_fe['ozone_grade'] = pd.cut(
        df_fe['hour_bef_ozone'],
        bins=[-np.inf, 0.03, 0.06, 0.1, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # 미세먼지 비율 (초미세먼지 비중)
    df_fe['pm_ratio'] = df_fe['hour_bef_pm2.5'] / (df_fe['hour_bef_pm10'] + 1)
    
    # 종합 대기질 등급 (최악 기준)
    df_fe['air_quality_worst'] = df_fe[['pm10_grade', 'pm25_grade', 'ozone_grade']].max(axis=1)
    
    # 대기질 지수 (가중 평균)
    df_fe['air_quality_index'] = (df_fe['hour_bef_pm10'] * 0.4 + 
                                  df_fe['hour_bef_pm2.5'] * 0.6)
    
    # ========== 기타 특성 ==========
    # 시정 그룹화
    df_fe['visibility_group'] = pd.cut(df_fe['hour_bef_visibility'], 
                                       bins=[0, 500, 1000, 1500, 2001],
                                       labels=[0, 1, 2, 3]).astype(int)
    
    # 온도 구간
    df_fe['temp_group'] = pd.cut(df_fe['hour_bef_temperature'],
                                 bins=[-10, 0, 10, 20, 30, 40],
                                 labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 상호작용 특성
    df_fe['temp_humidity'] = df_fe['hour_bef_temperature'] * df_fe['hour_bef_humidity']
    df_fe['temp_wind'] = df_fe['hour_bef_temperature'] * df_fe['hour_bef_windspeed']
    
    return df_fe

# 특성 엔지니어링 적용
train_fe = create_features(train_df)
test_fe = create_features(test_df)

print(f'특성 엔지니어링 후 Train 컬럼 수: {train_fe.shape[1]}')
print(f'특성 엔지니어링 후 Test 컬럼 수: {test_fe.shape[1]}')
print(f'\n새로 생성된 특성들:')
new_features = [col for col in train_fe.columns if col not in train_df.columns]
print(new_features)


# ============================================================
# [Cell 82] 학습/검증 데이터 분리 - 개선 버전
# (시계열 분할 + 로그 변환)
# ============================================================
# 기존 코드를 아래로 교체하세요

from autogluon.tabular import TabularPredictor

# 시간 순서대로 정렬 (id 기준 - 시간순으로 부여된 것으로 가정)
train_fe_sorted = train_fe.sort_values('id').reset_index(drop=True)

# 시계열 분할 (마지막 20%를 검증 데이터로)
split_idx = int(len(train_fe_sorted) * 0.8)

# 학습 데이터 준비
X = train_fe_sorted.drop(['id', 'count'], axis=1)
y = train_fe_sorted['count']

X_train = X.iloc[:split_idx]
X_val = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_val = y.iloc[split_idx:]

# 학습 데이터 결합
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)

# ★★★ 로그 변환 적용 ★★★
train_data['count'] = np.log1p(train_data['count'])
val_data['count'] = np.log1p(val_data['count'])

print(f'학습 데이터 크기: {train_data.shape}')
print(f'검증 데이터 크기: {val_data.shape}')
print(f'\n로그 변환 적용됨: count → log1p(count)')
print(f'변환 후 count 통계:\n{train_data["count"].describe()}')


# ============================================================
# [Cell 84] 모델 학습 - 개선 버전
# (스태킹 제어 + Early Stopping + Regularization)
# ============================================================
# 기존 코드를 아래로 교체하세요

import os
import shutil

# 기존 모델 폴더 삭제 (새로 학습)
model_path = './autogluon_models_v2'
if os.path.exists(model_path):
    shutil.rmtree(model_path)

np.random.seed(42)

# GPU 최적화 하이퍼파라미터
hyperparameters = {
    'GBM': {
        'learning_rate': 0.03,
        'num_leaves': 128,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
    },
    'CAT': {
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3,
    },
    'XGB': {
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    },
    'NN_TORCH': {
        'weight_decay': 1e-4,
        'dropout_prob': 0.2,
    },
    'RF': {'n_estimators': 300},
    'XT': {'n_estimators': 300},
}

# AutoGluon 모델 학습
predictor = TabularPredictor(
    label='count',
    problem_type='regression',
    eval_metric='root_mean_squared_error',
    path=model_path
)

# 모델 학습 (과적합 방지 설정 적용)
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    use_bag_holdout=True,
    time_limit=900,  # 15분
    presets='best_quality',
    
    # ★★★ 스태킹 레벨 제어 ★★★
    num_stack_levels=1,  # L2까지만 스태킹 (과적합 방지)
    num_bag_folds=5,     # 폴드 수 조정
    
    # 하이퍼파라미터 (정규화 포함)
    hyperparameters=hyperparameters,
    
    # GPU/CPU 설정
    num_gpus=1,
    num_cpus=16,
    
    verbosity=2
)

print("\n=== 모델 학습 완료 ===")


# ============================================================
# [Cell 85 - 신규 추가] 모델 리더보드 확인
# ============================================================

print('=== 학습된 모델 리더보드 ===')
leaderboard = predictor.leaderboard(val_data)
display(leaderboard)


# ============================================================
# [Cell 86 - 신규 추가] Feature Importance 분석
# ============================================================

import matplotlib.pyplot as plt

# Feature Importance 계산
print("=== Feature Importance 분석 ===")
importance = predictor.feature_importance(val_data)
print(importance)

# 시각화
plt.figure(figsize=(12, 10))
importance_sorted = importance.sort_values('importance', ascending=True)
plt.barh(range(len(importance_sorted)), importance_sorted['importance'], color='steelblue')
plt.yticks(range(len(importance_sorted)), importance_sorted.index)
plt.xlabel('Importance')
plt.title('Feature Importance (AutoGluon)')
plt.tight_layout()
plt.show()

# 저성능 특성 식별 (중요도 0 이하)
low_importance_features = importance[importance['importance'] <= 0].index.tolist()
print(f"\n제거 권장 특성 (중요도 ≤ 0): {low_importance_features}")


# ============================================================
# [Cell 87 - 신규 추가] 검증 데이터 평가 (로그 역변환 적용)
# ============================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 예측 (로그 스케일)
y_val_pred_log = predictor.predict(val_data.drop('count', axis=1))

# ★★★ 로그 역변환 ★★★
y_val_pred = np.expm1(y_val_pred_log)
y_val_actual = np.expm1(val_data['count'])

# 음수 예측값 처리
y_val_pred = np.maximum(y_val_pred, 0)

# 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
mae = mean_absolute_error(y_val_actual, y_val_pred)
r2 = r2_score(y_val_actual, y_val_pred)

print("=== 검증 데이터 평가 결과 (원본 스케일) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 실제 vs 예측
axes[0].scatter(y_val_actual, y_val_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
axes[0].plot([0, y_val_actual.max()], [0, y_val_actual.max()], 'r--', lw=2)
axes[0].set_xlabel('실제값')
axes[0].set_ylabel('예측값')
axes[0].set_title(f'실제 vs 예측 (RMSE: {rmse:.2f})')

# 잔차 분포
residuals = y_val_actual - y_val_pred
axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_xlabel('잔차 (실제 - 예측)')
axes[1].set_ylabel('빈도')
axes[1].set_title('잔차 분포')

plt.tight_layout()
plt.show()


# ============================================================
# [Cell 88 - 신규 추가] 테스트 데이터 예측 및 제출 파일 생성
# ============================================================

# 테스트 데이터 준비
test_features = test_fe.drop(['id'], axis=1)

# 예측 (로그 스케일)
test_pred_log = predictor.predict(test_features)

# ★★★ 로그 역변환 ★★★
test_pred = np.expm1(test_pred_log)

# 음수 예측값 처리
test_pred = np.maximum(test_pred, 0)

# 정수로 반올림 (대여량은 정수)
test_pred = np.round(test_pred).astype(int)

# 제출 파일 생성
submission = pd.DataFrame({
    'id': test_fe['id'],
    'count': test_pred
})

# 저장
submission_path = './submission_improved.csv'
submission.to_csv(submission_path, index=False)

print("=== 제출 파일 생성 완료 ===")
print(f"저장 경로: {submission_path}")
print(f"\n예측값 통계:\n{submission['count'].describe()}")
print(f"\n처음 10개 예측:")
print(submission.head(10))


# ============================================================
# [Cell 89 - 선택사항] 저성능 특성 제거 후 재학습
# ============================================================
# 필요시 주석 해제하여 사용

"""
# 저성능 특성 제거
if len(low_importance_features) > 0:
    print(f"제거할 특성: {low_importance_features}")
    
    train_data_filtered = train_data.drop(columns=low_importance_features, errors='ignore')
    val_data_filtered = val_data.drop(columns=low_importance_features, errors='ignore')
    
    # 새 모델 경로
    model_path_v3 = './autogluon_models_v3'
    if os.path.exists(model_path_v3):
        shutil.rmtree(model_path_v3)
    
    # 새 모델 학습
    predictor_v3 = TabularPredictor(
        label='count',
        problem_type='regression',
        eval_metric='root_mean_squared_error',
        path=model_path_v3
    )
    
    predictor_v3.fit(
        train_data=train_data_filtered,
        tuning_data=val_data_filtered,
        use_bag_holdout=True,
        time_limit=600,
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        hyperparameters=hyperparameters,
        num_gpus=1,
        num_cpus=16,
        verbosity=2
    )
    
    print("\n=== 특성 제거 후 재학습 완료 ===")
"""
