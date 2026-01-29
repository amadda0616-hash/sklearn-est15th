# ============================================================
# Pro_6 FIXED - 타이타닉 생존자 예측
# 수정사항:
# 1. Sex 특성 유지 (VIF에서 Sex_Pclass만 제거)
# 2. 앙상블에 SVC, GradientBoosting 추가
# 3. 검증 전략 개선 (전체 CV 사용)
# 4. Threshold 최적화
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import platform
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Korean Font Settings
system_name = platform.system()
if system_name == 'Windows':
    print('🪟 Windows: Malgun Gothic 설정')
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': 
    print('🍎 Mac: AppleGothic 설정')
    plt.rc('font', family='AppleGothic')
else:
    print('🐧 Linux/Other: NanumGothic 설정')
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False
print('✅ 라이브러리 임포트 및 한글 폰트 설정 완료')

# ============================================================
# 2. 데이터 불러오기
# ============================================================
base_path = r'C:/Users/user/github/DataScience/scikit-learn/scikit-learn/data/titanic'
train_df = pd.read_csv(f'{base_path}/train.csv')
test_df = pd.read_csv(f'{base_path}/test.csv')

test_passenger_ids = test_df['PassengerId'].copy()
train_len = len(train_df)
all_data = pd.concat([train_df, test_df], ignore_index=True)

print(f'Train: {train_df.shape}, Test: {test_df.shape}, All: {all_data.shape}')

# ============================================================
# 3. KNN Imputer for Age, Fare, Embarked
# ============================================================
def find_best_k_neighbors(train_df):
    print('🔍 KNN Imputer n_neighbors 최적화 중...')
    results = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)  # Reduced for speed
    
    for k in range(3, 12, 2):
        df = train_df.copy()
        df['Sex_num'] = (df['Sex'] == 'male').astype(int)
        df['Embarked_num'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(-1)
        
        imputer_cols = ['Pclass', 'Sex_num', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_num']
        imputer = KNNImputer(n_neighbors=k)
        df_imputed = pd.DataFrame(imputer.fit_transform(df[imputer_cols]), columns=imputer_cols)
        df['Age'] = df_imputed['Age']
        df['Fare'] = df_imputed['Fare']
        df['Embarked_num'] = df_imputed['Embarked_num'].round().astype(int)
        
        X = df[['Pclass', 'Sex_num', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_num']]
        y = df['Survived'].astype(int)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results.append({'k': k, 'cv_accuracy': scores.mean()})
        print(f'  k={k}: CV Accuracy = {scores.mean():.4f}')
    
    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df['cv_accuracy'].idxmax(), 'k']
    print(f'\n🏆 최적 n_neighbors = {best_k}')
    return int(best_k)

best_k = find_best_k_neighbors(train_df)

# ============================================================
# 4. WCG (Women, Children, Group) 전략 + KNN Imputer 적용
# ============================================================
def add_wcg_family_survival(all_data, train_len, best_k):
    all_data = all_data.copy()
    
    # Basic features
    all_data['Last_Name'] = all_data['Name'].apply(lambda x: x.split(',')[0])
    all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    all_data['TicketFrequency'] = all_data['Ticket'].map(all_data['Ticket'].value_counts())
    all_data['Sex_num'] = (all_data['Sex'] == 'male').astype(int)
    all_data['Embarked_num'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # KNN Imputer for Age, Fare, Embarked
    print(f'🔧 KNN Imputer 적용 (n_neighbors={best_k})...')
    imputer_cols = ['Pclass', 'Sex_num', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_num']
    imputer = KNNImputer(n_neighbors=best_k)
    df_imputed = pd.DataFrame(imputer.fit_transform(all_data[imputer_cols]), columns=imputer_cols, index=all_data.index)
    all_data['Age'] = df_imputed['Age']
    all_data['Fare'] = df_imputed['Fare']
    all_data['Embarked_num'] = df_imputed['Embarked_num'].round().astype(int)
    
    # Fare per person
    all_data['Fare_Per_Person'] = all_data['Fare'] / all_data['TicketFrequency']
    all_data['Fare_Per_Person_Round'] = all_data['Fare_Per_Person'].round(2)
    
    # Gender/Age flags
    all_data['IsChild'] = (all_data['Age'] < 10).astype(int)
    all_data['IsFemale'] = (all_data['Sex'] == 'female').astype(int)
    all_data['IsMaster'] = (all_data['Title'] == 'Master').astype(int)
    all_data['IsMaleChild'] = ((all_data['Sex'] == 'male') & (all_data['IsChild'] == 1)).astype(int)
    
    # Family Survival
    all_data['Family_Survival'] = 0.5
    all_data['WCG_Survival'] = 0
    
    print('📊 Family_Survival 계산 중...')
    for ticket, grp_df in all_data.groupby('Ticket'):
        if len(grp_df) > 1:
            for idx in grp_df.index:
                others = grp_df.drop(idx)
                others_train = others[others.index < train_len]
                if len(others_train) > 0:
                    if others_train['Survived'].max() == 1.0:
                        all_data.loc[idx, 'Family_Survival'] = 1
                    elif others_train['Survived'].min() == 0.0:
                        all_data.loc[idx, 'Family_Survival'] = 0
    
    for (last_name, fare_pp, embarked), grp_df in all_data.groupby(['Last_Name', 'Fare_Per_Person_Round', 'Embarked_num']):
        if len(grp_df) > 1:
            for idx in grp_df.index:
                if all_data.loc[idx, 'Family_Survival'] == 0.5:
                    others = grp_df.drop(idx)
                    others_train = others[others.index < train_len]
                    if len(others_train) > 0:
                        if others_train['Survived'].max() == 1.0:
                            all_data.loc[idx, 'Family_Survival'] = 1
                        elif others_train['Survived'].min() == 0.0:
                            all_data.loc[idx, 'Family_Survival'] = 0
    
    print('🎯 WCG 전략 적용 중...')
    for ticket, grp_df in all_data.groupby('Ticket'):
        if len(grp_df) > 1:
            grp_train = grp_df[grp_df.index < train_len]
            if len(grp_train) > 0:
                women_children = grp_train[(grp_train['IsFemale'] == 1) | (grp_train['IsChild'] == 1)]
                if len(women_children) > 0 and (women_children['Survived'] == 1).all():
                    for idx in grp_df.index:
                        if all_data.loc[idx, 'IsMaster'] == 1 or all_data.loc[idx, 'IsMaleChild'] == 1:
                            all_data.loc[idx, 'WCG_Survival'] = 1
                            all_data.loc[idx, 'Family_Survival'] = 1
    
    print(f'✅ Family_Survival 분포: {all_data["Family_Survival"].value_counts().to_dict()}')
    return all_data

all_data = add_wcg_family_survival(all_data, train_len, best_k)
print('\n✅ KNN Imputer + WCG 특성 생성 완료')

# ============================================================
# 5. Feature Engineering (Preserved from Pro_5)
# ============================================================
def preprocessing_pro5(all_data):
    all_data = all_data.copy()
    
    title_mapping = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Capt': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Jonkheer': 'Rare'}
    all_data['Title'] = all_data['Title'].map(title_mapping).fillna('Rare')
    all_data['Sex'] = (all_data['Sex'] == 'male').astype(int)
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
    
    def age_to_bin(age):
        if pd.isna(age): return 2
        elif age < 10: return 0
        elif age < 18: return 1
        elif age < 35: return 2
        elif age < 50: return 3
        else: return 4
    all_data['AgeBin'] = all_data['Age'].apply(age_to_bin).astype(int)
    all_data['LogFare'] = np.log1p(all_data['Fare_Per_Person'])
    all_data['Deck'] = all_data['Cabin'].str[0].fillna('U')
    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 0, 'U': 0}
    all_data['Deck'] = all_data['Deck'].map(deck_map).fillna(0).astype(int)
    title_enc = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    all_data['Title'] = all_data['Title'].map(title_enc).fillna(4).astype(int)
    all_data['Sex_Pclass'] = all_data['Sex'] * all_data['Pclass']
    
    return all_data

all_data = preprocessing_pro5(all_data)
print('✅ Pro5 Feature Engineering 완료 (Original Logic Preserved)')

# ============================================================
# 5.5 Pro_6 Additional Feature Engineering (Expanded Binning)
# ============================================================
def add_pro6_features(all_data):
    all_data = all_data.copy()
    
    # FareBin (New)
    all_data['FareBin'] = pd.qcut(
        all_data['Fare_Per_Person'].fillna(all_data['Fare_Per_Person'].median()), 
        q=5, labels=[0,1,2,3,4], duplicates='drop'
    ).astype(int)
    
    # FamilySizeBin (New)
    def familysize_to_bin(size):
        if size == 1: return 0      # Alone
        elif size <= 3: return 1     # Small family
        else: return 2                # Large family
    all_data['FamilySizeBin'] = all_data['FamilySize'].apply(familysize_to_bin).astype(int)
    
    return all_data

all_data = add_pro6_features(all_data)
print('✅ Pro6 추가 특성(FareBin, FamilySizeBin) 생성 완료')

# ============================================================
# 6. 데이터 분리 및 특성 리스트 정의 [FIXED: Include Sex]
# ============================================================

# [FIXED] Pro6 Features - Sex 유지!
features = ['Pclass', 'Sex', 'FareBin', 'Embarked_num', 'Title', 'FamilySizeBin', 
            'Family_Survival', 'TicketFrequency', 'Deck', 'AgeBin', 'Sex_Pclass']

print(f"📌 초기 분석 대상 특성 ({len(features)}개): {features}")

train_processed = all_data.iloc[:train_len].copy()
test_processed = all_data.iloc[train_len:].copy()

X = train_processed[features]
y = train_processed['Survived'].astype(int)

# [FIXED] Use full CV instead of small holdout split for more stable validation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_optuna = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'Full Dataset: {X.shape}')

# ============================================================
# 6.5 VIF Analysis [FIXED: Only remove Sex_Pclass, keep Sex]
# ============================================================
def calculate_vif(df, features_list):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_list
    vif_data["VIF"] = [variance_inflation_factor(df[features_list].values, i) 
                       for i in range(len(features_list))]
    return vif_data.sort_values('VIF', ascending=False)

print("\n🔍 VIF Analysis (다중공선성 검사):")
vif_df = calculate_vif(X, features)
print(vif_df)

# [FIXED] Only remove Sex_Pclass (interaction term), KEEP Sex as it's critical!
# Sex is THE most important feature for Titanic (~0.74 correlation with Survived)
high_vif_to_remove = ['Sex_Pclass']  # Only remove interaction term, NOT Sex
removed = [f for f in high_vif_to_remove if f in features]
if removed:
    print(f"\n⚠️ 교호작용항만 제거 (Sex는 필수 유지): {removed}")
    features = [f for f in features if f not in removed]
    
    # Re-calculate VIF to confirm
    X = X[features]
    print("✅ VIF 재검증 결과:")
    print(calculate_vif(X, features))

print(f"\n📌 VIF 필터링 후 최종 특성 ({len(features)}개): {features}")
print(f"🔥 Sex 특성 포함 여부: {'Sex' in features}")

# ============================================================
# 7. Baseline Models Performance (CV on full data)
# ============================================================
X_scaled = scaler.fit_transform(X)

baseline_models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'SVC': SVC(probability=True, random_state=42, class_weight='balanced'),
    'KNeighbors': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=1.6)
}

print('\n🚀 Baseline Models Performance (CV on Full Data):')
baseline_results = {}
for name, model in baseline_models.items():
    X_use = X_scaled if name in ['SVC', 'KNeighbors', 'LogisticRegression'] else X
    scores = cross_val_score(model, X_use, y, cv=cv, scoring='accuracy')
    baseline_results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f'  {name}: {scores.mean():.4f} ± {scores.std():.4f}')

# ============================================================
# 8. Optuna Hyperparameter Tuning
# ============================================================
class AdaptiveOverfittingMonitor:
    def __init__(self, max_depth_range=(3, 10), subsample_range=(0.6, 1.0), reg_alpha_min=1e-5, reg_lambda_min=1e-5):
        self.max_depth_range = list(max_depth_range)
        self.subsample_range = list(subsample_range)
        self.reg_alpha_min = reg_alpha_min
        self.reg_lambda_min = reg_lambda_min

    def adjust_for_overfitting(self):
        print("⚠️ 과적합 감지! 규제 강화 중...")
        if self.max_depth_range[1] > 3:
            self.max_depth_range[1] -= 1
        self.subsample_range[0] = min(0.9, self.subsample_range[0] + 0.05)
        self.reg_alpha_min *= 5
        self.reg_lambda_min *= 5
        print(f"  -> Max Depth <= {self.max_depth_range[1]}, Subsample >= {self.subsample_range[0]:.2f}")
        return True
    
    def reset(self):
        self.max_depth_range = [3, 10]
        self.subsample_range = [0.6, 1.0]

adaptive_params = AdaptiveOverfittingMonitor()

def check_overfitting_with_feedback(model, X_train, y_train, name, cv, threshold=0.03):
    train_pred = model.predict(X_train)
    train_score = accuracy_score(y_train, train_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_score = cv_scores.mean()
    
    gap = train_score - cv_score
    print(f"✅ {name}: Train={train_score:.4f}, CV={cv_score:.4f}, Gap={gap:.4f}")
    
    if gap > threshold:
        return True, train_score, cv_score, gap
    return False, train_score, cv_score, gap

# ============================================================
# 9. Tune XGBoost
# ============================================================
def tune_xgb(X_train, y_train, cv, cv_optuna, adaptive_params, max_iterations=2):
    adaptive_params.reset()
    for iteration in range(max_iterations):
        print(f'\n{"="*50}\n🚀 XGBoost 튜닝 #{iteration + 1}\n{"="*50}')
        def objective(trial):
            params = {
                'scale_pos_weight': 1.6,
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', adaptive_params.max_depth_range[0], adaptive_params.max_depth_range[1]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', adaptive_params.subsample_range[0], adaptive_params.subsample_range[1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', adaptive_params.reg_alpha_min, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', adaptive_params.reg_lambda_min, 10.0, log=True),
                'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1
            }
            return cross_val_score(XGBClassifier(**params), X_train, y_train, cv=cv_optuna, scoring='accuracy').mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42+iteration))
        study.optimize(objective, n_trials=30, timeout=120, show_progress_bar=True)
        print(f'Best CV: {study.best_value:.4f}')
        
        best_model = XGBClassifier(**study.best_params, scale_pos_weight=1.6, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        best_model.fit(X_train, y_train)
        
        is_overfitting, _, _, _ = check_overfitting_with_feedback(best_model, X_train, y_train, 'XGBoost', cv)
        if not is_overfitting: return study, best_model
        if not adaptive_params.adjust_for_overfitting(): return study, best_model
    return study, best_model

study_xgb, best_xgb = tune_xgb(X, y, cv, cv_optuna, adaptive_params)

# ============================================================
# 10. Tune RandomForest
# ============================================================
def tune_rf(X_train, y_train, cv, cv_optuna, adaptive_params, max_iterations=2):
    adaptive_params.reset()
    for iteration in range(max_iterations):
        print(f'\n{"="*50}\n🚀 RandomForest 튜닝 #{iteration + 1}\n{"="*50}')
        def objective(trial):
            params = {
                'class_weight': 'balanced',
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', adaptive_params.max_depth_range[0], adaptive_params.max_depth_range[1] + 2),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42, 'n_jobs': -1
            }
            return cross_val_score(RandomForestClassifier(**params), X_train, y_train, cv=cv_optuna, scoring='accuracy').mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42+iteration))
        study.optimize(objective, n_trials=30, timeout=120, show_progress_bar=True)
        print(f'Best CV: {study.best_value:.4f}')
        
        best_model = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)
        
        is_overfitting, _, _, _ = check_overfitting_with_feedback(best_model, X_train, y_train, 'RandomForest', cv)
        if not is_overfitting: return study, best_model
        if not adaptive_params.adjust_for_overfitting(): return study, best_model
    return study, best_model

study_rf, best_rf = tune_rf(X, y, cv, cv_optuna, adaptive_params)

# ============================================================
# 11. Tune SVC [NEW - Added to ensemble]
# ============================================================
def tune_svc(X_train, y_train, cv, cv_optuna):
    print(f'\n{"="*50}\n🚀 SVC 튜닝\n{"="*50}')
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        }
        return cross_val_score(SVC(**params), X_train, y_train, cv=cv_optuna, scoring='accuracy').mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=20, timeout=60, show_progress_bar=True)
    print(f'Best CV: {study.best_value:.4f}')
    
    best_model = SVC(**study.best_params, class_weight='balanced', probability=True, random_state=42)
    best_model.fit(X_train, y_train)
    
    _, _, cv_score, _ = check_overfitting_with_feedback(best_model, X_train, y_train, 'SVC', cv)
    return study, best_model

study_svc, best_svc = tune_svc(X_scaled, y, cv, cv_optuna)

# ============================================================
# 12. Tune GradientBoosting [NEW - Added to ensemble]
# ============================================================
def tune_gb(X_train, y_train, cv, cv_optuna):
    print(f'\n{"="*50}\n🚀 GradientBoosting 튜닝\n{"="*50}')
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        return cross_val_score(GradientBoostingClassifier(**params), X_train, y_train, cv=cv_optuna, scoring='accuracy').mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=30, timeout=120, show_progress_bar=True)
    print(f'Best CV: {study.best_value:.4f}')
    
    best_model = GradientBoostingClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    _, _, cv_score, _ = check_overfitting_with_feedback(best_model, X_train, y_train, 'GradientBoosting', cv)
    return study, best_model

study_gb, best_gb = tune_gb(X, y, cv, cv_optuna)

# ============================================================
# 13. Find Optimal Threshold via CV [NEW]
# ============================================================
print(f'\n{"="*50}\n🎯 Threshold 최적화\n{"="*50}')

# Use XGBoost for threshold optimization
probs = cross_val_predict(best_xgb, X, y, cv=cv, method='predict_proba')[:, 1]

best_threshold = 0.5
best_acc = 0
threshold_results = []
for t in np.arange(0.35, 0.65, 0.05):
    acc = accuracy_score(y, (probs >= t).astype(int))
    threshold_results.append({'threshold': t, 'accuracy': acc})
    print(f"  Threshold {t:.2f}: Accuracy = {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_threshold = t

print(f"\n🏆 최적 Threshold: {best_threshold:.2f} (Accuracy: {best_acc:.4f})")

# ============================================================
# 14. Ensemble (4 Models) [FIXED: Added SVC and GB]
# ============================================================
print(f'\n{"="*50}\n🔥 4-Model Ensemble\n{"="*50}')

# Create wrapper for SVC to use scaled features
class ScaledSVC(BaseEstimator):
    def __init__(self, svc_model, scaler):
        self.svc_model = svc_model
        self.scaler = scaler
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.svc_model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svc_model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.svc_model.predict_proba(X_scaled)

# Wrap SVC with scaler
scaled_svc = ScaledSVC(best_svc, StandardScaler())
scaled_svc.fit(X, y)

tuned_models = {
    'XGBoost': best_xgb,
    'RandomForest': best_rf,
    'SVC': scaled_svc,
    'GradientBoosting': best_gb
}

# Evaluate each tuned model using CV
print('\n📊 Tuned Models CV Performance:')
for name, model in tuned_models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'  {name}: {scores.mean():.4f} ± {scores.std():.4f}')

# Create ensembles
estimators = [(name, model) for name, model in tuned_models.items()]

try:
    # Soft Voting
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X, y)
    soft_cv_scores = cross_val_score(voting_soft, X, y, cv=cv, scoring='accuracy')
    print(f'\nVoting (Soft) CV: {soft_cv_scores.mean():.4f} ± {soft_cv_scores.std():.4f}')
    
    # Hard Voting
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X, y)
    hard_cv_scores = cross_val_score(voting_hard, X, y, cv=cv, scoring='accuracy')
    print(f'Voting (Hard) CV: {hard_cv_scores.mean():.4f} ± {hard_cv_scores.std():.4f}')
    
    # Stacking
    stacking = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'), 
        cv=5, n_jobs=-1
    )
    stacking.fit(X, y)
    stacking_cv_scores = cross_val_score(stacking, X, y, cv=cv, scoring='accuracy')
    print(f'Stacking CV: {stacking_cv_scores.mean():.4f} ± {stacking_cv_scores.std():.4f}')
    
    # Find best ensemble
    ensemble_results = {
        'Voting_Soft': soft_cv_scores.mean(),
        'Voting_Hard': hard_cv_scores.mean(),
        'Stacking': stacking_cv_scores.mean()
    }
    best_ensemble_name = max(ensemble_results, key=ensemble_results.get)
    print(f'\n🏆 Best Ensemble: {best_ensemble_name} (CV: {ensemble_results[best_ensemble_name]:.4f})')
    
except Exception as e:
    print(f'⚠️ Ensemble Error: {e}')

# ============================================================
# 15. 최종 제출 (Test Set Prediction)
# ============================================================
print(f'\n{"="*50}\n📝 최종 제출 파일 생성\n{"="*50}')

X_test_final = test_processed[features]
output_path = r'C:/Users/user/github/DataScience/scikit-learn/scikit-learn/Submission'
os.makedirs(output_path, exist_ok=True)

# Determine best model based on CV score
best_model_map = {
    'Voting_Soft': voting_soft,
    'Voting_Hard': voting_hard,
    'Stacking': stacking
}
best_model = best_model_map[best_ensemble_name]

# Generate submissions
for name, model, use_threshold in [
    ('Pro6_Fixed_Voting_Soft', voting_soft, True),
    ('Pro6_Fixed_Voting_Hard', voting_hard, False),
    ('Pro6_Fixed_Stacking', stacking, True),
    (f'Pro6_Fixed_Best_{best_ensemble_name}', best_model, True if best_ensemble_name != 'Voting_Hard' else False)
]:
    save_path = f'{output_path}/submission_{name}.csv'
    
    if use_threshold and hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test_final)[:, 1]
        pred = (probs >= best_threshold).astype(int)
        print(f'ℹ️ Applying Threshold {best_threshold:.2f} for {name}')
    else:
        pred = model.predict(X_test_final).astype(int)
        
    pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': pred}).to_csv(save_path, index=False)
    print(f'✅ {save_path} saved (Survived: {pred.sum()}/{len(pred)})')

print('\n🎉 Pro_6 FIXED - All Processes Completed!')
print(f'\n📊 Summary of Fixes Applied:')
print(f'  1. ✅ Sex 특성 복원 (VIF에서 Sex_Pclass만 제거)')
print(f'  2. ✅ 앙상블 확장 (4개 모델: XGBoost, RF, SVC, GB)')
print(f'  3. ✅ 검증 전략 개선 (전체 CV 사용)')
print(f'  4. ✅ Threshold 최적화 ({best_threshold:.2f})')
