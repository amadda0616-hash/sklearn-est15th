# ============================================================
# 수정 필요: 로그 변환 적용 시 모델 비교 코드도 업데이트 필요!
# ============================================================

"""
문제 원인:
- 모델은 log1p(count)로 학습됨
- 예측값도 로그 스케일로 출력됨
- 하지만 기존 비교 코드는 원본 스케일의 y_val과 비교 → RMSE 폭등

해결 방법:
- 예측값에 np.expm1() 적용하여 원본 스케일로 역변환
- 또는 y_val도 로그 스케일로 변환하여 비교
"""

# ============================================================
# [중요] y_val 변수 정의 추가 (Cell 82 아래에 추가 필요)
# ============================================================

# y_val을 원본 스케일로 저장해두어야 함
y_val_original = np.expm1(val_data['count'].copy())  # 역변환된 원본 값

print(f"y_val_original 통계 (원본 스케일):\n{y_val_original.describe()}")


# ============================================================
# [Cell 89/90] 개별 모델 성능 비교 - 수정 버전
# ★★★ 기존 코드를 아래로 완전히 교체 ★★★
# ============================================================

from sklearn.metrics import mean_squared_error

# 개별 모델 성능 비교 (로그 역변환 적용)
print('=== 개별 모델 성능 비교 (원본 스케일) ===')
model_results = {}

for model_name in predictor.model_names():
    try:
        # 예측 (로그 스케일)
        pred_log = predictor.predict(val_data.drop('count', axis=1), model=model_name)
        
        # ★★★ 로그 역변환 ★★★
        pred_original = np.expm1(pred_log)
        pred_original = np.maximum(pred_original, 0)  # 음수 방지
        
        # 원본 스케일에서 RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_val_original, pred_original))
        model_results[model_name] = rmse
    except Exception as e:
        print(f"  {model_name}: 에러 - {e}")

# 결과 정렬 및 출력
model_results_sorted = dict(sorted(model_results.items(), key=lambda x: x[1]))
for model, rmse in model_results_sorted.items():
    print(f'{model}: RMSE = {rmse:.4f}')


# ============================================================
# [Cell 91] 모델별 성능 시각화 - 수정 버전
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
models = list(model_results_sorted.keys())[:10]  # 상위 10개
rmses = [model_results_sorted[m] for m in models]

plt.barh(range(len(models)), rmses, color='steelblue', edgecolor='black')
plt.yticks(range(len(models)), models)
plt.xlabel('RMSE (낮을수록 좋음)')
plt.title('모델별 RMSE 비교 (상위 10개) - 원본 스케일')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# [추가 확인] 로그 변환이 제대로 되었는지 디버깅 코드
# ============================================================

print("\n=== 디버깅: 스케일 확인 ===")
print(f"val_data['count'] (로그 스케일):")
print(f"  min: {val_data['count'].min():.2f}, max: {val_data['count'].max():.2f}")
print(f"  (로그 스케일이면 보통 0~6 범위)")

print(f"\ny_val_original (원본 스케일):")
print(f"  min: {y_val_original.min():.2f}, max: {y_val_original.max():.2f}")
print(f"  (원본 스케일이면 보통 0~350 범위)")

# 예측값 확인
test_pred_log = predictor.predict(val_data.drop('count', axis=1).head(5))
test_pred_original = np.expm1(test_pred_log)
print(f"\n예측값 비교 (5개 샘플):")
print(f"  로그 스케일: {test_pred_log.values[:5]}")
print(f"  원본 스케일: {test_pred_original.values[:5]}")


# ============================================================
# 만약 로그 변환 없이 원래대로 돌리고 싶다면:
# Cell 82에서 이 두 줄을 주석 처리하세요:
# ============================================================
"""
# 주석 처리할 부분:
# train_data['count'] = np.log1p(train_data['count'])
# val_data['count'] = np.log1p(val_data['count'])
"""
