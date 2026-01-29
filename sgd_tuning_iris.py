
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 파이프라인 구축 (스케일링 필수)
# SGDClassifier는 특성 스케일에 매우 민감하므로 StandardScaler가 필수적입니다.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(random_state=42, n_jobs=-1))
])

# 4. 튜닝할 하이퍼파라미터 그리드 설정
param_grid = {
    'sgd__loss': ['hinge', 'log_loss', 'modified_huber', 'perceptron'], # hinge: SVM, log_loss: 로지스틱 회귀
    'sgd__penalty': ['l2', 'l1', 'elasticnet'],
    'sgd__alpha': [1e-4, 1e-3, 1e-2, 1e-1], # 규제 강도
    'sgd__learning_rate': ['optimal', 'adaptive', 'invscaling'],
    'sgd__eta0': [0.01, 0.1], # learning_rate가 adaptive일 때 중요
    'sgd__max_iter': [1000, 2000, 5000], # 충분한 반복 횟수
    'sgd__tol': [1e-3, 1e-4]
}

# 5. GridSearchCV 실행
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("튜닝 시작...")
grid_search.fit(X_train, y_train)

# 6. 결과 출력
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 교차 검증 점수: {grid_search.best_score_:.4f}")

# 7. 테스트 세트 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n테스트 세트 정확도: {test_acc:.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
