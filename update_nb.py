import json
import os
import sys

print("Starting update script...")
path = r'c:\Users\user\github\DataScience\scikit-learn\scikit-learn\Pro_5_sklearn_titanic_Survival_Han.ipynb'
log_path = r'c:\Users\user\github\DataScience\scikit-learn\scikit-learn\update_log.txt'

with open(log_path, 'w', encoding='utf-8') as log:
    log.write("Starting...\n")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        log.write(f"Loaded notebook (cells: {len(nb['cells'])}).\n")
    except Exception as e:
        log.write(f"Error loading: {e}\n")
        print(f"Error: {e}")
        sys.exit(1)

    # New content for Section 6
    new_section_6 = [
        "# ============================================================\n",
        "# 6. 데이터 분리\n",
        "# ============================================================\n",
        "\n",
        "# Feature Refinement: Removed High VIF features (Age, Fare_Per_Person, SibSp, Parch, IsAlone, IsMaleChild, IsMaster, WCG_Survival)\n",
        "features = ['Pclass', 'Sex', 'LogFare', 'Embarked_num', 'Title', 'FamilySize', 'IsChild', \n",
        "            'Family_Survival', 'TicketFrequency', 'Deck', 'AgeBin', 'Sex_Pclass']\n",
        "\n",
        "train_processed = all_data.iloc[:train_len].copy()\n",
        "test_processed = all_data.iloc[train_len:].copy()\n",
        "\n",
        "X = train_processed[features]\n",
        "y = train_processed['Survived'].astype(int)\n",
        "X_test = test_processed[features]\n",
        "\n",
        "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)\n",
        "\n",
        "cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)\n",
        "cv_optuna = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_val_scaled = scaler.transform(X_val)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "print(f'Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}')\n",
        "print(f'CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=10) → 총 50회')\n",
        "print(f'✅ StandardScaler fitted (for SVC, KNN, MLP, LogReg)')"
    ]

    # New content for Section 7-4
    new_section_7_4 = [
        "# ============================================================\n",
        "# 7-4. VIF (Variance Inflation Factor) 검증 (Feature Refinement 후)\n",
        "# ============================================================\n",
        "print('📊 VIF (Variance Inflation Factor) 검증')\n",
        "print('   * 다중공선성 제거 후 남은 Feature들의 VIF 확인')\n",
        "print('   VIF > 10: 여전히 심각한 다중공선성 존재 가능성\\n')\n",
        "\n",
        "def calculate_vif(X):\n",
        "    \"\"\"Calculate VIF for all features\"\"\"\n",
        "    vif_data = pd.DataFrame()\n",
        "    vif_data['Feature'] = X.columns\n",
        "    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n",
        "    return vif_data.sort_values('VIF', ascending=False)\n",
        "\n",
        "vif_df = calculate_vif(X_train)\n",
        "vif_df['Status'] = vif_df['VIF'].apply(\n",
        "    lambda x: '🚨 주의' if x > 10 else ('⚠️ 경고' if x > 5 else '✅ 정상')\n",
        ")\n",
        "\n",
        "print(vif_df.to_string(index=False))\n",
        "\n",
        "# Visualization\n",
        "plt.figure(figsize=(12, 8))\n",
        "colors = ['#e74c3c' if v > 10 else '#f1c40f' if v > 5 else '#2ecc71' for v in vif_df['VIF']]\n",
        "plt.barh(vif_df['Feature'], vif_df['VIF'], color=colors)\n",
        "plt.axvline(x=5, color='orange', linestyle='--', label='경고 임계값 (VIF=5)')\n",
        "plt.axvline(x=10, color='red', linestyle='--', label='심각 임계값 (VIF=10)')\n",
        "plt.xlabel('VIF')\n",
        "plt.title('VIF Verification - Feature Refinement 후 검증', fontsize=14)\n",
        "plt.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('\\n✅ VIF 검증 완료 (Multicollinear Feature 제거됨)')"
    ]

    found_6 = False
    found_7_4 = False

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "# 6. 데이터 분리" in source_str:
                cell['source'] = new_section_6
                found_6 = True
                log.write(f"Updated Section 6 at cell {i}\n")
            elif "# 7-4. VIF" in source_str or "VIF (Variance Inflation Factor) 분석" in source_str:
                cell['source'] = new_section_7_4
                found_7_4 = True
                log.write(f"Updated Section 7-4 at cell {i}\n")

    if found_6:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=4)
            log.write("SUCCESS: Saved notebook.\n")
            print("SUCCESS: Notebook updated.")
        except Exception as e:
            log.write(f"Error saving: {e}\n")
            print(f"Error saving: {e}")
    else:
        log.write(f"FAILURE: Found6={found_6}, Found74={found_7_4}\n")
        print("FAILURE: Cells not found.")
