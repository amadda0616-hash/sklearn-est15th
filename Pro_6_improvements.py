# ============================================================
# Pro_6 Improvement Code - DATA LEAK FREE VERSION
# ============================================================
# Key fixes:
# 1. StandardScaler inside Pipeline (no leak)
# 2. Fixed feature list (no RFECV leak)
# 3. VIF analysis for reference only, not for feature selection
# ============================================================


# ============================================================
# CELL 5 - Feature Engineering (ADD FareBin, FamilySizeBin)
# OVERWRITE existing Cell 5
# ============================================================
"""
# ============================================================
# 5. Feature Engineering (with Expanded Binning)
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
    
    # AgeBin
    def age_to_bin(age):
        if pd.isna(age): return 2
        elif age < 10: return 0
        elif age < 18: return 1
        elif age < 35: return 2
        elif age < 50: return 3
        else: return 4
    all_data['AgeBin'] = all_data['Age'].apply(age_to_bin).astype(int)
    
    # FareBin (NEW)
    def fare_to_bin(fare):
        if pd.isna(fare) or fare <= 7.91: return 0
        elif fare <= 14.45: return 1
        elif fare <= 31.0: return 2
        elif fare <= 100.0: return 3
        else: return 4
    all_data['FareBin'] = all_data['Fare_Per_Person'].apply(fare_to_bin).astype(int)
    
    # FamilySizeBin (NEW)
    def familysize_to_bin(size):
        if size == 1: return 0
        elif size <= 3: return 1
        elif size <= 5: return 2
        else: return 3
    all_data['FamilySizeBin'] = all_data['FamilySize'].apply(familysize_to_bin).astype(int)
    
    all_data['LogFare'] = np.log1p(all_data['Fare_Per_Person'])
    all_data['Deck'] = all_data['Cabin'].str[0].fillna('U')
    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 0, 'U': 0}
    all_data['Deck'] = all_data['Deck'].map(deck_map).fillna(0).astype(int)
    title_enc = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    all_data['Title'] = all_data['Title'].map(title_enc).fillna(4).astype(int)
    all_data['Sex_Pclass'] = all_data['Sex'] * all_data['Pclass']
    
    return all_data

all_data = preprocessing_pro5(all_data)
print('✅ Feature Engineering 완료 (FareBin, FamilySizeBin 추가)')
"""


# ============================================================
# CELL 5.5 - VIF Analysis (REFERENCE ONLY, NO FEATURE REMOVAL)
# INSERT after Cell 5
# ============================================================
"""
# ============================================================
# 5.5 VIF Analysis (Reference Only - No Data Leak)
# ============================================================

# Fixed feature list (no dynamic selection to avoid leak)
features = ['Pclass', 'Sex', 'LogFare', 'Embarked_num', 'Title', 
            'Family_Survival', 'TicketFrequency', 'Deck', 'AgeBin', 
            'FareBin', 'FamilySizeBin']
# Note: Sex_Pclass removed due to high VIF with Sex

train_processed = all_data.iloc[:train_len].copy()
test_processed = all_data.iloc[train_len:].copy()

# VIF Analysis for REFERENCE ONLY
print('📊 VIF Analysis (Reference Only):')
print('=' * 50)

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

X_temp = train_processed[features]
vif_df = calculate_vif(X_temp)
print(vif_df.to_string(index=False))
print(f'\\n✅ Using fixed feature list ({len(features)} features) - no leak from VIF')
"""


# ============================================================
# CELL 6 - Data Split (NO SCALER LEAK)
# OVERWRITE existing Cell 6
# ============================================================
"""
# ============================================================
# 6. 데이터 분리 (No Data Leak)
# ============================================================

# Use fixed features (no RFECV to avoid leak)
X = train_processed[features]
y = train_processed['Survived'].astype(int)
X_test = test_processed[features]

# CV settings
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_optuna = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

# NOTE: No scaler.fit_transform() here! 
# Scaler will be inside Pipeline to prevent leak

print(f'Train: {X.shape}, Test: {X_test.shape}')
print(f'Features ({len(features)}): {features}')
print(f'✅ No StandardScaler leak - will use Pipeline')
"""


# ============================================================
# CELL 10 - Model Comparison (WITH PIPELINE - NO LEAK)
# OVERWRITE existing Cell 10
# ============================================================
"""
# ============================================================
# 10. Model Comparison (No Leak - Using Pipelines)
# ============================================================

# All models that need scaling use Pipeline
tree_models = {
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'GaussianNB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis()
}

# Scale-sensitive models MUST use Pipeline
scale_models = {
    'SVC': Pipeline([('scaler', StandardScaler()), ('model', SVC(probability=True, random_state=42))]),
    'KNeighbors': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_jobs=-1))]),
    'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))]),
    'MLP': Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier(max_iter=1000, random_state=42))])
}

all_models = {**tree_models, **scale_models}

print('📊 Model Comparison (No Leak):')
model_scores = {}
for name, model in all_models.items():
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        model_scores[name] = scores.mean()
        print(f'  {name}: {scores.mean():.4f} ± {scores.std():.4f}')
    except Exception as e:
        print(f'  {name}: Error - {e}')

sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
print(f'\\n🏆 Top 5:')
for name, score in sorted_models[:5]: print(f'  {name}: {score:.4f}')
"""


# ============================================================
# CELL 11 - XGBoost Tuning (NO LEAK)
# OVERWRITE existing Cell 11
# ============================================================
"""
# ============================================================
# 11. Optuna XGBoost Tuning (No Leak)
# ============================================================

def tune_xgb(X_data, y_data, cv, cv_optuna, adaptive_params, max_iterations=3):
    adaptive_params.reset()
    for iteration in range(max_iterations):
        print(f'\\n{"="*50}\\n🚀 XGBoost 튜닝 #{iteration + 1}\\n{"="*50}')
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250),
                'max_depth': trial.suggest_int('max_depth', adaptive_params.max_depth_range[0], adaptive_params.max_depth_range[1]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', adaptive_params.subsample_range[0], adaptive_params.subsample_range[1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', adaptive_params.reg_alpha_min, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', adaptive_params.reg_lambda_min, 10.0, log=True),
                'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1
            }
            return cross_val_score(XGBClassifier(**params), X_data, y_data, cv=cv_optuna, scoring='accuracy').mean()
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42+iteration))
        study.optimize(objective, n_trials=25, timeout=90, show_progress_bar=True)
        print(f'Best CV: {study.best_value:.4f}')
        best_model = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        best_model.fit(X_data, y_data)
        is_overfitting, _, _, _ = check_overfitting_with_feedback(best_model, X_data, y_data, 'XGBoost', cv)
        if not is_overfitting: return study, best_model
        if not adaptive_params.adjust_for_overfitting(): return study, best_model
    return study, best_model

study_xgb, best_xgb = tune_xgb(X, y, cv, cv_optuna, adaptive_params)
"""


# ============================================================
# CELL 12 - RandomForest Tuning (NO LEAK)
# OVERWRITE existing Cell 12
# ============================================================
"""
# ============================================================
# 12. Optuna RandomForest Tuning (No Leak)
# ============================================================

def tune_rf(X_data, y_data, cv, cv_optuna, adaptive_params, max_iterations=3):
    adaptive_params.reset()
    for iteration in range(max_iterations):
        print(f'\\n{"="*50}\\n🚀 RandomForest 튜닝 #{iteration + 1}\\n{"="*50}')
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250),
                'max_depth': trial.suggest_int('max_depth', adaptive_params.max_depth_range[0], adaptive_params.max_depth_range[1] + 3),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'random_state': 42, 'n_jobs': -1
            }
            return cross_val_score(RandomForestClassifier(**params), X_data, y_data, cv=cv_optuna, scoring='accuracy').mean()
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42+iteration))
        study.optimize(objective, n_trials=25, timeout=90, show_progress_bar=True)
        print(f'Best CV: {study.best_value:.4f}')
        best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
        best_model.fit(X_data, y_data)
        is_overfitting, _, _, _ = check_overfitting_with_feedback(best_model, X_data, y_data, 'RandomForest', cv)
        if not is_overfitting: return study, best_model
        if not adaptive_params.adjust_for_overfitting(): return study, best_model
    return study, best_model

study_rf, best_rf = tune_rf(X, y, cv, cv_optuna, adaptive_params)
"""


# ============================================================
# CELL 13 - GradientBoosting Tuning (NO LEAK)
# OVERWRITE existing Cell 13
# ============================================================
"""
# ============================================================
# 13. Optuna GradientBoosting Tuning (No Leak)
# ============================================================

def tune_gb(X_data, y_data, cv, cv_optuna, adaptive_params, max_iterations=3):
    adaptive_params.reset()
    for iteration in range(max_iterations):
        print(f'\\n{"="*50}\\n🚀 GradientBoosting 튜닝 #{iteration + 1}\\n{"="*50}')
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 250),
                'max_depth': trial.suggest_int('max_depth', adaptive_params.max_depth_range[0], adaptive_params.max_depth_range[1]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', adaptive_params.subsample_range[0], adaptive_params.subsample_range[1]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'random_state': 42
            }
            return cross_val_score(GradientBoostingClassifier(**params), X_data, y_data, cv=cv_optuna, scoring='accuracy').mean()
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42+iteration))
        study.optimize(objective, n_trials=25, timeout=90, show_progress_bar=True)
        print(f'Best CV: {study.best_value:.4f}')
        best_model = GradientBoostingClassifier(**study.best_params, random_state=42)
        best_model.fit(X_data, y_data)
        is_overfitting, _, _, _ = check_overfitting_with_feedback(best_model, X_data, y_data, 'GradientBoosting', cv)
        if not is_overfitting: return study, best_model
        if not adaptive_params.adjust_for_overfitting(): return study, best_model
    return study, best_model

study_gb, best_gb = tune_gb(X, y, cv, cv_optuna, adaptive_params)
"""


# ============================================================
# CELL 13.5 - SVC Tuning (WITH PIPELINE - NO LEAK)
# INSERT after Cell 13
# ============================================================
"""
# ============================================================
# 13.5 Optuna SVC Tuning (Pipeline - No Leak)
# ============================================================

def tune_svc(X_data, y_data, cv, cv_optuna):
    print(f'\\n{"="*50}\\n🚀 SVC 튜닝 #1\\n{"="*50}')
    
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
            'probability': True,
            'random_state': 42
        }
        # Use Pipeline to prevent leak!
        pipe = Pipeline([('scaler', StandardScaler()), ('model', SVC(**params))])
        return cross_val_score(pipe, X_data, y_data, cv=cv_optuna, scoring='accuracy').mean()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=25, timeout=90, show_progress_bar=True)
    print(f'Best CV: {study.best_value:.4f}')
    
    # Final model with Pipeline
    best_model = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', SVC(**study.best_params, probability=True, random_state=42))
    ])
    best_model.fit(X_data, y_data)
    
    cv_score = cross_val_score(best_model, X_data, y_data, cv=cv, scoring='accuracy').mean()
    print(f'✅ SVC Pipeline CV: {cv_score:.4f}')
    
    return study, best_model

study_svc, best_svc = tune_svc(X, y, cv, cv_optuna)
"""


# ============================================================
# CELL 14 - CV Performance
# OVERWRITE existing Cell 14
# ============================================================
"""
# ============================================================
# 14. CV Performance
# ============================================================

tuned_models = {'XGBoost': best_xgb, 'RandomForest': best_rf, 'GradientBoosting': best_gb, 'SVC': best_svc}

print('📊 Final CV Performance:')
cv_results = {}
for name, model in tuned_models.items():
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_results[name] = cv_scores.mean()
    print(f'  {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
"""


# ============================================================
# CELL 15 - Ensemble (NO LEAK)
# OVERWRITE existing Cell 15
# ============================================================
"""
# ============================================================
# 15. Ensemble (No Leak)
# ============================================================

estimators = [
    ('XGBoost', best_xgb),
    ('RandomForest', best_rf),
    ('GradientBoosting', best_gb),
    ('SVC', best_svc)
]

try:
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X, y)
    voting_soft_cv = cross_val_score(voting_soft, X, y, cv=cv, scoring='accuracy').mean()
    print(f'Voting (Soft) CV: {voting_soft_cv:.4f}')

    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X, y)
    voting_hard_cv = cross_val_score(voting_hard, X, y, cv=cv, scoring='accuracy').mean()
    print(f'Voting (Hard) CV: {voting_hard_cv:.4f}')
    
    stacking = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(C=0.1, max_iter=1000), 
        cv=5, n_jobs=-1
    )
    stacking.fit(X, y)
    stacking_cv = cross_val_score(stacking, X, y, cv=cv, scoring='accuracy').mean()
    print(f'Stacking CV: {stacking_cv:.4f}')
    
    print(f'\\n🏆 Best Ensemble CV: {max(voting_soft_cv, voting_hard_cv, stacking_cv):.4f}')

except Exception as e:
    print(f'⚠️ Ensemble Error: {e}')
"""


# ============================================================
# CELL 16 - Final Submission
# OVERWRITE existing Cell 16
# ============================================================
"""
# ============================================================
# 16. 최종 제출
# ============================================================
import os

output_path = r'C:/Users/user/github/DataScience/scikit-learn/scikit-learn/Submission'
os.makedirs(output_path, exist_ok=True)

for name, model in [
    ('Pro6_Voting_Soft', voting_soft),
    ('Pro6_Voting_Hard', voting_hard),
    ('Pro6_Stacking', stacking)
]:
    pred = model.predict(X_test).astype(int)
    pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': pred}).to_csv(
        f'{output_path}/submission_{name}.csv', index=False)
    print(f'✅ submission_{name}.csv saved ({pred.sum()}/{len(pred)})')

# Best single model
best_single_name = max(cv_results, key=cv_results.get)
best_single_model = tuned_models[best_single_name]
pred = best_single_model.predict(X_test).astype(int)
pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': pred}).to_csv(
    f'{output_path}/submission_Pro6_{best_single_name}.csv', index=False)
print(f'✅ submission_Pro6_{best_single_name}.csv saved ({pred.sum()}/{len(pred)})')

print('\\n🎉 Complete!')
print('\\n📋 Data Leak Prevention Summary:')
print('  - Fixed feature list (no RFECV leak)')
print('  - StandardScaler inside Pipeline (no scaling leak)')
print('  - VIF for reference only (no selection leak)')
"""
