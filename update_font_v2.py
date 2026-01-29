import json
import os
import sys

path = r'c:\Users\user\github\DataScience\scikit-learn\scikit-learn\Pro_5_sklearn_titanic_Survival_Han.ipynb'
log_path = r'c:\Users\user\github\DataScience\scikit-learn\scikit-learn\font_update_log.txt'

print(f"Reading {path}...")
with open(log_path, 'w', encoding='utf-8') as log:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        log.write("Loaded notebook successfully.\n")
    except Exception as e:
        log.write(f"Error loading notebook: {e}\n")
        print(f"Error: {e}")
        sys.exit(1)

    # New content for Section 1
    new_section_1 = [
        "# ============================================================\n",
        "# 1. 라이브러리 임포트\n",
        "# ============================================================\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import sys\n",
        "import platform\n",
        "\n",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score\n",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
        "from sklearn.impute import KNNImputer\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.base import BaseEstimator, TransformerMixin, clone\n",
        "from sklearn.pipeline import Pipeline\n",
        "\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
        "from xgboost import XGBClassifier\n",
        "\n",
        "# VIF calculation\n",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
        "\n",
        "import optuna\n",
        "from optuna.samplers import TPESampler\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
        "\n",
        "# Visualization settings\n",
        "%config InlineBackend.figure_format = 'retina'\n",
        "sns.set_style('whitegrid')\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "plt.rcParams['axes.labelsize'] = 12\n",
        "plt.rcParams['axes.titlesize'] = 14\n",
        "\n",
        "# ------------------------------------------------------------\n",
        "# Korean Font Settings\n",
        "# ------------------------------------------------------------\n",
        "system_name = platform.system()\n",
        "if system_name == 'Windows':\n",
        "    print('🪟 Windows: Malgun Gothic 설정')\n",
        "    plt.rc('font', family='Malgun Gothic')\n",
        "elif system_name == 'Darwin': \n",
        "    print('🍎 Mac: AppleGothic 설정')\n",
        "    plt.rc('font', family='AppleGothic')\n",
        "elif 'google.colab' in sys.modules:\n",
        "    print('☁️ Colab: NanumBarunGothic 설치 및 설정')\n",
        "    !apt-get update -qq\n",
        "    !apt-get install fonts-nanum* -qq\n",
        "    import matplotlib.font_manager as fm\n",
        "    fe = fm.FontEntry(\n",
        "        fname=r'/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',\n",
        "        name='NanumBarunGothic')\n",
        "    fm.fontManager.ttflist.insert(0, fe)\n",
        "    plt.rc('font', family=fe.name)\n",
        "else:\n",
        "    # Linux / Kaggle / Other\n",
        "    print('🐧 Linux/Other: NanumGothic 설정')\n",
        "    plt.rc('font', family='NanumGothic')\n",
        "\n",
        "plt.rcParams['axes.unicode_minus'] = False\n",
        "\n",
        "print('✅ 라이브러리 임포트 및 한글 폰트 설정 완료')"
    ]

    found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "# 1. 라이브러리 임포트" in source_str or "import pandas as pd" in source_str:
                cell['source'] = new_section_1
                found = True
                log.write(f"Updated Section 1 at cell index {i}\n")
                print("Updated Section 1 with font settings")
                break

    if found:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=4)
            print("SUCCESS: Notebook updated.")
            log.write("SUCCESS: Saved notebook.\n")
        except Exception as e:
            log.write(f"Error saving notebook: {e}\n")
            print(f"Error saving: {e}")
    else:
        print("FAILURE: Section 1 not found.")
        log.write("FAILURE: Section 1 not found.\n")
