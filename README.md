# Scikit-Learn Machine Learning Repository (sklearn-est15th)

이 저장소는 Scikit-Learn을 활용한 머신러닝 학습, 데이터 분석, 그리고 다양한 모델링 실습 코드를 포함하고 있습니다.
기초적인 데이터 전처리부터 고급 앙상블 기법, 하이퍼파라미터 튜닝(Optuna), 그리고 웹 애플리케이션 배포까지 다루고 있습니다.

## 📂 폴더 및 파일 구조

### 📚 학습 노트 (Curriculum Notebooks)
Scikit-Learn의 핵심 기능을 순차적으로 학습하는 노트북들입니다.
- **기초 및 전처리**: `1_sklearn_start.ipynb`, `4_sklearn_PreProcess.ipynb`, `8_polynominal_Feature.ipynb`
- **모델링 및 알고리즘**:
    - 분류 (Classification): `3_SVM.ipynb`, `5_sklearn_classification.ipynb`
    - 회귀 (Regression): `9_LinearRegressionModel.ipynb`
    - 모델 선택 및 평가: `2_ModelSelection.ipynb`
    - 비지도 학습: `13_unsupervisedLearning.ipynb`
- **고급 기법**:
    - 앙상블 (Ensemble): `10_ensemble.ipynb`
    - 하이퍼파라미터 튜닝 (Optuna): `6_classification_Optuna.ipynb`, `11_ensemble_Optuna.ipynb`

### 🚀 프로젝트 및 실습 (Projects & Exercises)
실제 데이터를 활용한 분석 및 예측 프로젝트입니다.

#### Titanic 생존자 예측 (Titanic Survival Prediction)
- 다양한 접근 방식을 통한 타이타닉 생존자 예측 모델링
- 주요 파일: `7_Titanic.ipynb`, `Pro_1` ~ `Pro_6` 시리즈
- 특징: 데이터 전처리, 파생 변수 생성, 앙상블 모델링, Kaggle 제출용 파일 생성 (`Submission/`)

#### 와인 품질 분석 (Wine Quality Analysis)
- 레드 와인 품질 분류 및 회귀 분석
- 주요 파일: `Pro_3_sklearn_Red Wine.ipynb`, `Plus_1`, `Plus_4` 시리즈

#### 기타 분석
- **California Housing**: 지리 정보 시각화 및 가격 예측
- **AutoML**: AutoGluon 등을 활용한 자동화된 모델 학습 (`AutoML/`, `Plus_7_ensemble_gemini.ipynb`)

### 🌐 웹 애플리케이션 (Web Engineering)
- 머신러닝 모델을 웹 서비스로 배포하기 위한 코드
- 위치: `webML/` (Flask, FastAPI, Gradio 등을 활용한 예제 포함 예상)

## 🛠 사용 기술 (Tech Stack)
- **Language**: Python
- **Libraries**:
    - Machine Learning: Scikit-Learn, XGBoost, LightGBM, CatBoost, AutoGluon
    - Data Manipulation: Pandas, NumPy
    - Visualization: Matplotlib, Seaborn, Folium
    - Optimization: Optuna
    - Web: Flask, FastAPI (or Gradio)

## 📝 활용 방법
각 노트북(`.ipynb`)은 독립적인 주제를 다루거나 시리즈로 연결되어 있습니다. Jupyter Notebook 또는 Jupyter Lab 환경에서 실행할 수 있습니다.

```bash
# 필수 라이브러리 설치 예시
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm catboost optuna
```

---
*이 저장소는 머신러닝 실습 및 학습 기록을 위해 생성되었습니다.*
