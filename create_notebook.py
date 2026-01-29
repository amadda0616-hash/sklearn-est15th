import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# 1. Title & Data Description
nb.cells.append(new_markdown_cell("""# 타이타닉 생존자 예측 (Titanic Survival Prediction)

본 노트북은 Kaggle의 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) 데이터를 기반으로 생존자를 예측하는 모델을 구축합니다.

## 데이터셋 설명
- **Survival**: 0 = 사망, 1 = 생존
- **Pclass**: 티켓 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석)
- **Sex**: 성별
- **Age**: 나이
- **SibSp**: 함께 탑승한 형제자매, 배우자 수
- **Parch**: 함께 탑승한 부모, 자녀 수
- **Ticket**: 티켓 번호
- **Fare**: 요금
- **Cabin**: 객실 번호
- **Embarked**: 탑승 항구 (C = Cherbourg, Q = Queenstown, S = Southampton)

**분석 목표:**
1. 데이터 전처리 및 탐색적 데이터 분석 (EDA)
2. 다양한 머신러닝 모델 학습 (최신 고성능 모델 포함 12종)
3. 앙상블 기법 적용 (Voting, Stacking)
4. SMOTE를 이용한 데이터 불균형 처리 효과 분석
5. 하이퍼파라미터 튜닝을 통한 성능 최적화
"""))

# 2. Setup & Imports
nb.cells.append(new_code_cell("""# 필요한 라이브러리 설치 (실행 환경에 없는 경우 주석 해제 후 실행)
# !pip install xgboost lightgbm catboost plotly seaborn datasist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 모델링 라이브러리
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# 머신러닝 모델
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 최신 고성능 모델
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 불균형 데이터 처리
from imblearn.over_sampling import SMOTE

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows 환경)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

print("라이브러리 임포트 및 설정 완료")"""))

# 3. Data Loading
nb.cells.append(new_code_cell("""# 데이터 불러오기
# 경로 설정 (사용자 경로에 맞게 수정됨)
base_path = r'C:\Users\user\github\DataScience\scikit-learn\scikit-learn\data\titanic'
train_path = f'{base_path}/train.csv'
test_path = f'{base_path}/test.csv'

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("데이터 로딩 성공!")
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")
except FileNotFoundError:
    print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

# 데이터 상위 5개 행 확인
train_df.head()"""))

# 4. Data Preprocessing
nb.cells.append(new_markdown_cell("""## 4. 데이터 전처리
요청하신 대로 다음 작업을 수행합니다:
1. 'Ticket', 'PassengerId' 컬럼 삭제
2. 'Sex', 'Embarked' 등 명목형 변수 숫자 변환
3. 결측치 처리 (Median/Mean 대치, Cabin 제외)
"""))

nb.cells.append(new_code_cell("""# 전처리 함수 정의
def preprocess_data(df, is_train=True):
    df_copy = df.copy()
    
    # 1. 불필요한 컬럼 삭제 (PassengerId, Ticket)
    # 분석에서 제외할 Cabin 컬럼도 삭제 (결측치가 너무 많음)
    drop_cols = ['PassengerId', 'Ticket', 'Cabin']
    df_copy = df_copy.drop(columns=drop_cols, errors='ignore')
    print(f"삭제된 컬럼: {drop_cols}")

    # 2. 결측치 처리
    # Age: Median 값으로 대치
    df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].median())
    # Fare: Median 값으로 대치
    df_copy['Fare'] = df_copy['Fare'].fillna(df_copy['Fare'].median())
    # Embarked: 최빈값(Mode)으로 대치
    df_copy['Embarked'] = df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0])
    
    print("결측치 처리 완료 (Age/Fare -> Median, Embarked -> Mode)")

    # 3. 명목형 변수 숫자 변환 (Label Encoding)
    # Sex 변환
    le_sex = LabelEncoder()
    df_copy['Sex'] = le_sex.fit_transform(df_copy['Sex'])
    # 주석: Sex 변환 (male=1, female=0 등으로 변환됨 - 실제 매핑 확인 필요)
    print(f"Sex 인코딩 클래스: {le_sex.classes_}")

    # Embarked 변환
    le_emb = LabelEncoder()
    df_copy['Embarked'] = le_emb.fit_transform(df_copy['Embarked'])
    # 주석: Embarked 변환 (C=0, Q=1, S=2)
    print(f"Embarked 인코딩 클래스: {le_emb.classes_}")
    
    # Name 컬럼에서 Title 추출 후 변환 (선택 사항 - 성능 향상을 위해 추가)
    df_copy['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, 
        "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
        "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3
    }
    df_copy['Title'] = df_copy['Title'].map(title_mapping)
    df_copy['Title'] = df_copy['Title'].fillna(3) # 기타
    df_copy = df_copy.drop('Name', axis=1)
    
    return df_copy

# 전처리 적용
print("--- Train Data 전처리 ---")
train_processed = preprocess_data(train_df)
print("\\n--- Test Data 전처리 ---")
test_processed = preprocess_data(test_df, is_train=False)

# 결과 확인
print("\\n전처리 후 데이터 정보:")
train_processed.info()"""))

# 5. EDA & Visualization (Plotly)
nb.cells.append(new_markdown_cell("""## 5. 탐색적 데이터 분석 (EDA) 및 시각화
가독성이 좋고 색상 활용도가 높은 Plotly(인터랙티브)와 Seaborn을 사용하여 데이터를 분석합니다.
"""))

nb.cells.append(new_code_cell("""# 5.1 생존자 비율 시각화 (Pie Chart)
fig = px.pie(train_df, names='Survived', title='전체 생존자 비율', 
             color_discrete_sequence=px.colors.sequential.RdBu,
             hole=0.4)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

# 5.2 성별에 따른 생존율 (Bar Chart)
fig = px.histogram(train_df, x="Sex", color="Survived", barmode="group",
                   title="성별에 따른 생존 여부 분포",
                   color_discrete_sequence=['#EF553B', '#636EFA'], # Plotly 기본 색상
                   category_orders={"Survived": [0, 1]})
fig.update_layout(bargap=0.2)
fig.show()

# 5.3 객실 등급(Pclass)별 생존율
fig = px.histogram(train_df, x="Pclass", color="Survived", barmode="group",
                   title="객실 등급(Pclass)별 생존 여부",
                   color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()

# 5.4 나이 분포 및 생존 여부 (Histogram)
fig = px.histogram(train_df, x="Age", color="Survived", nbins=40,
                   title="나이 분포에 따른 생존 여부",
                   marginal="box", # 상단에 박스플롯 추가
                   opacity=0.7,
                   color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_layout(barmode="overlay")
fig.update_traces(opacity=0.75)
fig.show()

# 5.5 상관관계 히트맵 (Seaborn)
plt.figure(figsize=(12, 10))
plt.title("변수 간 상관관계 히트맵 (Processed Data)", fontsize=15)
corr_matrix = train_processed.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', linewidths=0.5, fmt='.2f')
plt.show()"""))

# 6. Feature Engineering
nb.cells.append(new_code_cell("""## 6. 특성 엔지니어링 (Feature Engineering)
가족 규모(FamilySize) 변수를 생성하고, 혼자 탑승했는지 여부(IsAlone)를 추가합니다.

# 가족 수 = 형제자매/배우자 + 부모/자녀 + 본인(1)
train_processed['FamilySize'] = train_processed['SibSp'] + train_processed['Parch'] + 1
test_processed['FamilySize'] = test_processed['SibSp'] + test_processed['Parch'] + 1

# 혼자 탑승 여부
train_processed['IsAlone'] = 1
train_processed.loc[train_processed['FamilySize'] > 1, 'IsAlone'] = 0

test_processed['IsAlone'] = 1
test_processed.loc[test_processed['FamilySize'] > 1, 'IsAlone'] = 0

print("특성 엔지니어링 완료: FamilySize, IsAlone 컬럼 추가됨")
train_processed.head()"""))

# 7. Model Modeling (12 Models)
nb.cells.append(new_markdown_cell("""## 7. 모델링 (Modeling)
총 12가지의 다양한 모델을 사용하여 학습을 진행합니다. 최신 고성능 모델인 XGBoost, LightGBM, CatBoost 등을 포함합니다.

### 사용할 모델 리스트:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Extra Trees**
5. **Gradient Boosting**
6. **AdaBoost**
7. **XGBoost**
8. **LightGBM**
9. **CatBoost**
10. **Histogram-based Gradient Boosting**
11. **Support Vector Machine (SVM)**
12. **K-Nearest Neighbors (KNN)**
"""))

nb.cells.append(new_code_cell("""# 학습 데이터셋 준비
X = train_processed.drop('Survived', axis=1)
y = train_processed['Survived']

# 학습/검증 데이터 분리 (8:2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42, algorithm='SAMME'), # 'SAMME' 지정 권장
    'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier()
}

print(f"총 {len(models)}개의 모델이 정의되었습니다.")"""))

# 8. Model Training & Selection
nb.cells.append(new_code_cell("""# 모델 학습 및 성능 평가
results = []

print("모델 학습 시작...")
for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # 평가 (Accuracy)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    # 결과 저장
    results.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Validation Accuracy': val_acc
    })
    print(f"{name} 완료 - Val Acc: {val_acc:.4f}")

# 결과 데이터프레임 생성 및 정렬
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Validation Accuracy', ascending=False)

# 결과 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x='Validation Accuracy', y='Model', data=results_df, palette='viridis')
plt.title('모델별 검증 정확도 비교', fontsize=15)
plt.xlim(0.7, 1.0) # 차이 부각을 위해 x축 조정
plt.grid(axis='x', alpha=0.5)
plt.show()

display(results_df)

# 상위 4개 모델 선정
top_4_models = results_df.head(4)['Model'].tolist()
print(f"\\n선정된 상위 4개 모델: {top_4_models}")"""))

# 9. Ensemble Learning
nb.cells.append(new_markdown_cell("""## 8. 앙상블 (Ensemble)
상위 4개 모델을 조합하여 앙상블 모델을 생성합니다.
1. **Hard Voting**: 다수결
2. **Soft Voting**: 확률의 평균
3. **Stacking**: 메타 모델을 이용한 재학습
"""))

nb.cells.append(new_code_cell("""# 상위 4개 모델 객체 가져오기
selected_estimators = [(name, models[name]) for name in top_4_models]

print(f"앙상블에 사용할 모델: {[m[0] for m in selected_estimators]}")

# 1. Hard Voting
hard_voting = VotingClassifier(estimators=selected_estimators, voting='hard')
hard_voting.fit(X_train, y_train)
hard_acc = accuracy_score(y_val, hard_voting.predict(X_val))

# 2. Soft Voting
soft_voting = VotingClassifier(estimators=selected_estimators, voting='soft')
soft_voting.fit(X_train, y_train)
soft_acc = accuracy_score(y_val, soft_voting.predict(X_val))

# 3. Stacking Ensemble
# 메타 모델로는 로지스틱 회귀나 랜덤 포레스트 등을 사용. 여기서는 LogisticRegression 사용
stacking_clf = StackingClassifier(
    estimators=selected_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X_train, y_train)
stacking_acc = accuracy_score(y_val, stacking_clf.predict(X_val))

print(f"Hard Voting Accuracy: {hard_acc:.4f}")
print(f"Soft Voting Accuracy: {soft_acc:.4f}")
print(f"Stacking Ensemble Accuracy: {stacking_acc:.4f}")

# 간단한 비교 그래프
ensemble_results = pd.DataFrame({
    'Method': ['Hard Voting', 'Soft Voting', 'Stacking'],
    'Accuracy': [hard_acc, soft_acc, stacking_acc]
})
plt.figure(figsize=(8, 4))
sns.barplot(x='Method', y='Accuracy', data=ensemble_results, palette='magma')
plt.title('앙상블 기법별 정확도 비교')
plt.ylim(0.75, 0.90)
plt.show()"""))

# 10. Helper Parameter Tuning
nb.cells.append(new_code_cell("""## 9. 하이퍼파라미터 튜닝
가장 성능이 좋았던 모델 하나를 선정하거나 Stacking 모델을 대상으로 하이퍼파라미터 튜닝을 진행할 수 있습니다.
여기서는 예시로 **Random Forest** 모델에 대해 튜닝을 진행해보겠습니다.
"""))

nb.cells.append(new_code_cell("""# Random Forest 하이퍼파라미터 튜닝 (GridSearchCV)
from sklearn.model_selection import GridSearchCV

# 튜닝할 파라미터 그리드 정의
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 기본 모델
rf = RandomForestClassifier(random_state=42)

# GridSearch 수행
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=1, scoring='accuracy')

grid_search.fit(X_train, y_train)

print(f"최적의 파라미터: {grid_search.best_params_}")
print(f"최고 교차 검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 검증 세트 평가
best_rf = grid_search.best_estimator_
tuned_acc = accuracy_score(y_val, best_rf.predict(X_val))
print(f"튜닝된 모델 Validation Accuracy: {tuned_acc:.4f}")"""))

# 11. SMOTE Comparison
nb.cells.append(new_markdown_cell("""## 11. SMOTE 적용에 따른 성능 비교 (AUC)
데이터 불균형 문제 해결을 위해 SMOTE(Synthetic Minority Over-sampling Technique)를 적용했을 때 성능(AUC)이 향상되는지 비교합니다.
"""))

nb.cells.append(new_code_cell("""# SMOTE 적용
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"SMOTE 적용 전 데이터 shape: {X_train.shape}")
print(f"SMOTE 적용 후 데이터 shape: {X_train_res.shape}")
print(f"Target 분포 (전):\\n{y_train.value_counts()}")
print(f"Target 분포 (후):\\n{y_train_res.value_counts()}")

# 비교를 위한 기본 모델 (Logistic Regression 사용)
# 1. 원본 데이터 학습
model_orig = LogisticRegression(max_iter=1000, random_state=42)
model_orig.fit(X_train, y_train)
pred_prob_orig = model_orig.predict_proba(X_val)[:, 1]
auc_orig = roc_auc_score(y_val, pred_prob_orig)

# 2. SMOTE 데이터 학습
model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_res, y_train_res)
pred_prob_smote = model_smote.predict_proba(X_val)[:, 1]
auc_smote = roc_auc_score(y_val, pred_prob_smote)

print(f"\\nOriginal Data AUC: {auc_orig:.4f}")
print(f"SMOTE Data AUC: {auc_smote:.4f}")

# AUC 비교 그래프
comparison_df = pd.DataFrame({
    'Data Type': ['Original', 'SMOTE'],
    'AUC Score': [auc_orig, auc_smote]
})

plt.figure(figsize=(6, 4))
sns.barplot(x='Data Type', y='AUC Score', data=comparison_df, palette='coolwarm')
plt.title('SMOTE 적용 여부에 따른 AUC 점수 비교')
plt.ylim(0.5, 1.0)
plt.show()

if auc_smote > auc_orig:
    print("결과: SMOTE 적용이 모델의 AUC 성능을 향상시켰습니다.")
else:
    print("결과: 이 데이터셋/모델에서는 SMOTE 적용이 큰 효과가 없거나 성능이 떨어졌습니다.")"""))

# 12. Final Evaluation & Submission
nb.cells.append(new_markdown_cell("""## 12. 결론 및 모델 튜닝 가능 수치 정리

### 모델 평가
- 다양한 모델 중 **Tree 기반 모델(XGBoost, Random Forest 등)**이 일반적으로 좋은 성능을 보입니다.
- **앙상블 기법(Stacking/Voting)**을 사용하면 단일 모델보다 안정적인 예측 성능을 얻을 수 있습니다.

### 튜닝 가능한 주요 파라미터 (xgboost, lightgbm 기준)
1. **learning_rate**: 학습률 (0.01 ~ 0.3). 낮을수록 세밀하게 학습하지만 시간 오래 걸림.
2. **n_estimators**: 트리의 개수. 너무 많으면 과적합 가능성.
3. **max_depth**: 트리의 깊이. 깊을수록 복잡한 모델 -> 과적합 주의.
4. **min_child_weight**: 리프 노드에 필요한 최소 가중치 합. 과적합 제어.
5. **subsample**: 각 트리 학습 시 사용할 데이터 샘플링 비율.
6. **colsample_bytree**: 각 트리 학습 시 사용할 특성(컬럼) 샘플링 비율.
7. **scale_pos_weight**: 불균형 데이터셋에서 양성 클래스의 가중치 조절 (SMOTE 대용으로 사용 가능).

이 노트북을 바탕으로 더 많은 특성 공학과 정밀한 튜닝을 수행하면 더 높은 점수를 얻을 수 있습니다.
"""))

# 파일 저장
output_filename = 'Pro_1_sklearn_titanic_Survial_Han.ipynb'
with open(output_filename, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"노트북 생성 완료: {output_filename}")
