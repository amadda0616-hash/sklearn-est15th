import json
import shutil
import os

# Updated to use relative path since the script is now in the same directory
source_file = "a-data-science-framework-to-achieve-99-accuracy_#translated.ipynb"
temp_file = "temp_notebook.ipynb"
log_file = "final_log.txt"

korean_step5 = [
    "<a id=\"ch7\"></a>\n",
    "# 5단계: 데이터 모델링\n",
    "데이터 과학은 수학(통계, 선형 대수 등), 컴퓨터 과학(프로그래밍 언어, 컴퓨터 시스템 등), 경영 관리(커뮤니케이션, 주제 관련 지식 등) 간의 다학제적 분야입니다. 대부분의 데이터 과학자는 이 세 가지 분야 중 하나에서 왔기 때문에 해당 분야로 기울어지는 경향이 있습니다. 그러나 데이터 과학은 세 다리가 있는 의자와 같아서 어느 한 다리가 다른 다리보다 더 중요하지 않습니다. 따라서 이 단계에서는 수학에 대한 고급 지식이 필요합니다. 하지만 걱정하지 마세요. 이 커널에서 다룰 높은 수준의 개요만 필요합니다. 또한 컴퓨터 과학 덕분에 많은 힘든 작업이 이미 수행되어 있습니다. 따라서 한때 수학이나 통계학 대학원 학위가 필요했던 문제도 이제는 몇 줄의 코드로 해결할 수 있습니다. 마지막으로 문제를 생각하려면 비즈니스 감각이 필요합니다. 결국 시각 장애인 안내견을 훈련시키는 것처럼 기계는 우리에게서 배우는 것이지 그 반대가 아닙니다.\n",
    "\n",
    "기계 학습(ML)은 이름에서 알 수 있듯이 기계에게 무엇을 생각해야 하는지가 아니라 어떻게 생각해야 하는지를 가르치는 것입니다. 이 주제와 빅 데이터는 수십 년 동안 존재해 왔지만 기업과 전문가 모두에게 진입 장벽이 낮아졌기 때문에 그 어느 때보다 인기가 높아지고 있습니다. 이것은 장단점이 있습니다. 장점은 이러한 알고리즘이 이제 현실 세계의 더 많은 문제를 해결할 수 있는 더 많은 사람들에게 접근 가능하다는 것입니다. 단점은 진입 장벽이 낮다는 것은 자신이 사용하는 도구를 모르는 사람들이 더 많아져 잘못된 결론에 도달할 수 있다는 것을 의미합니다. 그래서 저는 무엇을 해야 하는지뿐만 아니라 왜 해야 하는지를 가르치는 데 중점을 둡니다. 이전에 저는 누군가에게 십자 드라이버를 건네달라고 부탁했는데 일자 드라이버나 최악의 경우 망치를 건네주는 비유를 사용했습니다. 기껏해야 이해가 완전히 부족함을 보여줍니다. 최악의 경우 프로젝트 완료를 불가능하게 만들거나 더 나아가 잘못된 실행 가능한 정보를 구현하게 됩니다. 이제 요점을 강조했으므로(말장난 의도 없음), 무엇을 해야 하는지, 그리고 가장 중요한 **이유**를 보여 드리겠습니다.\n",
    "\n",
    "먼저 기계 학습의 목적은 인간의 문제를 해결하는 것임을 이해해야 합니다. 기계 학습은 지도 학습, 비지도 학습, 강화 학습으로 분류할 수 있습니다. 지도 학습은 정답이 포함된 훈련 데이터 세트를 제시하여 모델을 훈련하는 곳입니다. 비지도 학습은 정답이 포함되지 않은 훈련 데이터 세트를 사용하여 모델을 훈련하는 곳입니다. 그리고 강화 학습은 이전 두 가지의 하이브리드로, 모델에 정답이 즉시 제공되지 않고 학습을 강화하기 위해 일련의 이벤트 후에 나중에 제공됩니다. 우리는 기능 세트와 해당 타겟을 제시하여 알고리즘을 훈련하고 있기 때문에 지도 기계 학습을 수행하고 있습니다. 그런 다음 동일한 데이터 세트의 새 하위 세트를 제시하고 예측 정확도에서 유사한 결과를 얻기를 바랍니다.\n",
    "\n",
    "많은 기계 학습 알고리즘이 있지만 타겟 변수 및 데이터 모델링 목표에 따라 분류, 회귀, 클러스터링 또는 차원 축소의 네 가지 범주로 줄일 수 있습니다. 클러스터링과 차원 축소는 나중을 위해 남겨두고 분류와 회귀에 집중하겠습니다. 연속 타겟 변수에는 회귀 알고리즘이 필요하고 이산 타겟 변수에는 분류 알고리즘이 필요하다고 일반화할 수 있습니다. 한 가지 참고할 점은 로지스틱 회귀는 이름에 회귀가 있지만 실제로는 분류 알고리즘이라는 것입니다. 우리의 문제는 승객이 생존했는지 생존하지 못했는지를 예측하는 것이므로 이것은 이산 타겟 변수입니다. 분석을 시작하기 위해 *sklearn* 라이브러리의 분류 알고리즘을 사용할 것입니다. 다음 섹션에서 설명할 교차 검증 및 채점 지표를 사용하여 알고리즘의 성능을 순위를 매기고 비교할 것입니다.\n",
    "\n",
    "**기계 학습 선택:**\n",
    "* [Sklearn Estimator 개요](http://scikit-learn.org/stable/user_guide.html)\n",
    "* [Sklearn Estimator 세부 정보](http://scikit-learn.org/stable/modules/classes.html)\n",
    "* [Estimator 선택 마인드맵](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)\n",
    "* [Estimator 선택 치트 시트](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf)\n",
    "\n",
    "\n",
    "이제 솔루션을 지도 학습 분류 알고리즘으로 식별했습니다. 선택 목록을 좁힐 수 있습니다.\n",
    "\n",
    "**기계 학습 분류 알고리즘:**\n",
    "* [앙상블 방법](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)\n",
    "* [일반화 선형 모델 (GLM)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)\n",
    "* [나이브 베이즈](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)\n",
    "* [최근접 이웃](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)\n",
    "* [서포트 벡터 머신 (SVM)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)\n",
    "* [결정 트리](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)\n",
    "* [판별 분석](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)\n",
    "\n",
    "\n",
    "### 데이터 과학 101: 기계 학습 알고리즘 (MLA) 선택 방법\n",
    "**중요:** 데이터 모델링과 관련하여 초보자의 질문은 항상 \"가장 좋은 기계 학습 알고리즘은 무엇입니까?\"입니다. 이에 대해 초보자는 기계 학습의 [공짜 점심 없음 정리(NFLT)](http://robertmarks.org/Classes/ENGR5358/Papers/NFL_4_Dummies.pdf)를 배워야 합니다. 요컨대 NFLT는 모든 데이터 세트에 대해 모든 상황에서 가장 잘 작동하는 슈퍼 알고리즘은 없다는 것입니다. 따라서 가장 좋은 접근 방식은 여러 MLA를 시도하고 조정하고 특정 시나리오에 대해 비교하는 것입니다. 그렇긴 하지만 [Caruana & Niculescu-Mizil 2006](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf) (MLA 비교에 대한 [비디오 강의 여기](http://videolectures.net/solomon_caruana_wslmw/) 시청), 게놈 선택을 위해 NIH에서 수행한 [Ogutu et al. 2011](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3103196/), 17개 계열의 179개 분류기를 비교한 [Fernandez-Delgado et al. 2014](http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf), [Thoma 2016 sklearn 비교](https://martin-thoma.com/comparing-classifiers/)와 같이 알고리즘을 비교하기 위한 좋은 연구가 수행되었습니다. 또한 [더 많은 데이터가 더 나은 알고리즘을 이긴다](https://www.kdnuggets.com/2015/06/machine-learning-more-data-better-algorithms.html)는 학파도 있습니다. \n",
    "\n",
    "그렇다면 이 모든 정보로 초보자는 어디서부터 시작해야 할까요? [트리, 배깅, 랜덤 포레스트 및 부스팅](http://jessica2.msri.org/attachments/10778/10778-boost.pdf)으로 시작하는 것이 좋습니다. 이들은 기본적으로 결정 트리의 다른 구현이며, 배우고 이해하기 가장 쉬운 개념입니다. 또한 SVC와 같은 것보다 다음 섹션에서 설명할 조정이 더 쉽습니다. 아래에서는 여러 MLA를 실행하고 비교하는 방법에 대한 개요를 제공하지만 이 커널의 나머지 부분에서는 결정 트리 및 파생물을 통한 데이터 모델링 학습에 중점을 둘 것입니다."
]

korean_step51 = [
    "<a id=\"ch8\"></a>\n",
    "## 5.1 모델 성능 평가\n",
    "요약하자면, 몇 가지 기본적인 데이터 정리, 분석 및 기계 학습 알고리즘(MLA)을 통해 승객 생존을 약 82%의 정확도로 예측할 수 있습니다. 몇 줄의 코드치고는 나쁘지 않습니다. 하지만 우리가 항상 묻는 질문은 더 잘할 수 있는가, 그리고 더 중요한 것은 투자한 시간에 대해 ROI(투자 수익)를 얻을 수 있는가 하는 것입니다. 예를 들어 정확도를 0.1%만 높일 수 있다면 3개월 동안 개발할 가치가 있을까요? 연구 분야에 종사한다면 답은 '예'일 수 있지만 비즈니스 분야에 종사한다면 대답은 대부분 '아니요'입니다. 따라서 모델을 개선할 때 이 점을 명심하세요.\n",
    "\n",
    "### 데이터 과학 101: 기본 정확도 결정 ###\n",
    "모델을 개선하는 방법을 결정하기 전에 모델을 유지할 가치가 있는지 확인해 보겠습니다. 그러기 위해서는 데이터 과학 101의 기본으로 돌아가야 합니다. 결과가 두 가지뿐이기 때문에 이것이 이진 문제라는 것을 알고 있습니다. 승객은 생존했거나 사망했습니다. 동전 던지기 문제처럼 생각해보세요. 공정한 동전이 있고 앞면이나 뒷면을 추측하면 맞출 확률이 50-50입니다. 따라서 50%를 최악의 모델 성능으로 설정해 보겠습니다. 그보다 낮다면 동전을 던지면 되는데 왜 모델이 필요하겠습니까?\n",
    "\n",
    "좋습니다. 데이터 세트에 대한 정보가 없더라도 이진 문제에서는 항상 50%를 얻을 수 있습니다. 하지만 우리는 데이터 세트에 대한 정보가 있으므로 더 잘할 수 있어야 합니다. 우리는 2,224명 중 1,502명, 즉 67.5%의 사람들이 사망했다는 것을 알고 있습니다. 따라서 가장 빈번하게 발생하는 사건, 즉 100%의 사람들이 사망했다고 예측한다면 우리는 67.5%의 확률로 맞을 것입니다. 따라서 68%를 나쁜 모델 성능으로 설정해 보겠습니다. 그보다 낮다면 가장 빈번한 발생을 사용하여 예측할 수 있는데 왜 모델이 필요하겠습니까?\n",
    "\n",
    "### 데이터 과학 101: 나만의 모델 만드는 방법 ###\n",
    "정확도는 높아지고 있지만 더 잘할 수 있을까요? 데이터에 신호가 있나요? 이를 설명하기 위해 가장 개념화하기 쉽고 간단한 덧셈과 곱셈 계산이 필요한 자체 결정 트리 모델을 만들어 보겠습니다. 결정 트리를 만들 때는 타겟 응답을 분할하여 생존/1과 사망/0을 동질적인 하위 그룹으로 배치하는 질문을 하고 싶을 것입니다. 이것은 과학이자 예술이므로 21가지 질문 게임(스무고개)을 통해 어떻게 작동하는지 보여드리겠습니다. 스스로 따라하고 싶다면 훈련 데이터 세트를 다운로드하여 Excel로 가져오세요. 행에 아래 설명된 기능, 열에 생존, 값에 개수 및 행 개수 %를 사용하여 피벗 테이블을 만듭니다.\n",
    "\n",
    "게임의 이름은 결정 트리 모델을 사용하여 하위 그룹을 만들어 생존/1을 한 버킷에, 사망/0을 다른 버킷에 넣는 것입니다. 경험 법칙은 다수결입니다. 즉, 대다수 또는 50% 이상이 생존했다면 하위 그룹의 모든 사람이 생존/1했다고 가정하고, 50% 이하가 생존했다면 하위 그룹의 모든 사람이 사망/0했다고 가정합니다. 또한 하위 그룹이 10명 미만이거나 모델 정확도가 정체되거나 감소하면 중단합니다. 이해하셨나요? 시작해 봅시다!\n",
    "\n",
    "***질문 1: 타이타닉호에 타고 있었습니까?*** 예라면 대다수(62%)가 사망했습니다. 표본 생존율은 모집단의 68%와 다릅니다. 그럼에도 불구하고 모두 사망했다고 가정하면 표본 정확도는 62%입니다.\n",
    "\n",
    "***질문 2: 남성입니까 여성입니까?*** 남성, 대다수(81%) 사망. 여성, 대다수(74%) 생존. 정확도는 79%입니다.\n",
    "\n",
    "***질문 3A (여성 분기로 이동, 개수 = 314): 1, 2, 3등석 중 어디에 있습니까?*** 1등석, 대다수(97%) 생존, 2등석, 대다수(92%) 생존. 사망 하위 그룹이 10명 미만이므로 이 분기 진행을 중단합니다. 3등석은 50-50으로 균등합니다. 모델을 개선하기 위한 새로운 정보는 얻지 못했습니다.\n",
    "\n",
    "***질문 4A (여성 3등석 분기로 이동, 개수 = 144): C, Q, S 항구 중 어디에서 승선했습니까?*** 약간의 정보를 얻습니다. C와 Q는 여전히 대다수가 생존했으므로 변경 사항이 없습니다. 또한 사망 하위 그룹이 10명 미만이므로 중단합니다. S는 대다수(63%)가 사망했습니다. 따라서 여성, 3등석, S 승선인 경우 생존했다고 가정하던 것에서 사망했다고 가정하는 것으로 변경합니다. 모델 정확도가 81%로 증가합니다. \n",
    "\n",
    "***질문 5A (여성 3등석 S 승선 분기로 이동, 개수 = 88):*** 지금까지는 좋은 결정을 내린 것 같습니다. 다른 레벨을 추가해도 더 많은 정보를 얻을 수 없을 것 같습니다. 이 하위 그룹 55명은 사망했고 33명은 생존했습니다. 대다수가 사망했으므로 33명을 식별하거나 사망에서 생존으로 변경하여 모델 정확도를 높일 하위 그룹을 찾기 위한 신호가 필요합니다. 기능을 사용하여 놀아볼 수 있습니다. 제가 찾은 것 중 하나는 요금 0-8로, 대다수가 생존했습니다. 샘플 크기는 11-9로 작지만 통계에서 자주 사용됩니다. 정확도는 약간 향상되지만 82%를 넘기기에는 충분하지 않습니다. 그래서 여기서 멈추겠습니다.\n",
    "\n",
    "***질문 3B (남성 분기로 이동, 개수 = 577):*** 질문 2로 돌아가서 남성의 대다수가 사망했다는 것을 알고 있습니다. 따라서 대다수가 생존한 하위 그룹을 식별하는 기능을 찾고 있습니다. 놀랍게도 등급이나 승선지는 여성의 경우처럼 중요하지 않았지만 직함(Title)은 중요했으며 82%에 도달하게 해줍니다. 다른 기능을 추측하고 확인해도 82%를 넘는 것은 없는 것 같습니다. 그래서 지금은 여기서 멈추겠습니다.\n",
    "\n",
    "해냈습니다. 매우 적은 정보로 82%의 정확도에 도달했습니다. 최악, 나쁨, 좋음, 더 좋음, 최고 척도에서 82%는 괜찮은 결과를 제공하는 간단한 모델이므로 좋음으로 설정하겠습니다. 하지만 여전히 질문이 남습니다. 우리가 만든 모델보다 더 잘할 수 있을까요? \n",
    "\n",
    "그렇게 하기 전에 방금 위에서 쓴 내용을 코딩해 보겠습니다. 이것은 \"손\"으로 만든 수동 프로세스라는 점에 유의하세요. 이렇게 할 필요는 없지만 MLA 작업을 시작하기 전에 이해하는 것이 중요합니다. MLA를 미적분 시험의 TI-89 계산기라고 생각하세요. 매우 강력하고 많은 힘든 작업을 도와줍니다. 하지만 시험에서 무엇을 하고 있는지 모른다면 계산기, 심지어 TI-89라도 합격하는 데 도움이 되지 않을 것입니다. 그러니 다음 섹션을 현명하게 공부하세요.\n",
    "\n",
    "참조: [교차 검증 및 결정 트리 튜토리얼](http://www.cs.utoronto.ca/~fidler/teaching/2015/slides/CSC411/tutorial3_CrossVal-DTs.pdf)"
]

log = []

try:
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    print(f"Copying {source_file} to {temp_file}...")
    shutil.copy2(source_file, temp_file)
    
    with open(temp_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changed = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            
            # Loose match for Step 5
            if "ch7" in source:
                cell['source'] = korean_step5
                changed += 1
                
            # Loose match for ch8
            elif "ch8" in source:
                cell['source'] = korean_step51
                changed += 1
    
    if changed > 0:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"Moving {temp_file} back to {source_file}...")
        # Since move might default to copy+del, we use copy then del to be explicit
        if os.path.exists(source_file):
            os.remove(source_file)
        shutil.move(temp_file, source_file)
        log.append(f"Success! Updated {changed} cells.")
    else:
        log.append("No changes needed - condition not met.")

except Exception as e:
    message = f"Error: {e}"
    print(message)
    log.append(message)
    # Cleanup temp
    if os.path.exists(temp_file):
        os.remove(temp_file)

with open(log_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(log))
