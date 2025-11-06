# 🎬 Movie Recommendation System

**자동화된 딥러닝 추천 시스템 프로젝트**  
Movielens 1M 데이터를 기반으로 **AutoInt 모델**을 학습하고, Streamlit을 통해 사용자 맞춤 영화 추천을 제공합니다.

<br>

## 👀 주요 기능
✅ **AutoInt 모델 기반 추천**
- Self-Attention과 DNN을 활용한 Feature Interaction 학습  
- Embedding 및 Attention Layer 구조 확장으로 정확도 향상
   
✅ **정확도 시각화**
- Precision@K, AUC, Loss 그래프 자동 저장 및 시각화
  
✅ **사용자별 맞춤 추천**
- 입력 사용자에 대해 미시청 영화 중 상위 N개 추천
- 선호 장르/연대 기반 설명(`선호 장르(액션)`, `선호 연대(1990s)` 등)

<br>

## 🧩 프로젝트 구조
movie-recommendation-system/   
│   
├─ data/   
│ ├─ field_dims.npy   
│ ├─ label_encoders.pkl   
│ └─ ml-1m/   
│ ├─ movies_prepro.csv   
│ ├─ ratings_prepro.csv   
│ └─ users_prepro.csv   
│   
├─ model/   
│ ├─ autoInt_model.h5 # 레거시 HDF5 (by_name 로드용)   
│ ├─ autoInt_model.weights.h5 # Keras3 포맷 (fallback)   
│ └─ metrics.json # 학습 로그(AUC, Precision@K 등)   
│   
├─ train_autoint_optimized.py # 모델 학습 코드   
├─ show_st.py # Streamlit 서비스 코드   
└─ requirements.txt   

<br>

## ⚙️ 환경 설정
tensorflow==2.15.0.post1   
scikit-learn==1.3.2   
streamlit==1.39.0   
pandas==2.2.2   
numpy==1.26.4   
joblib==1.3.2   

<br>

## 🚨 모델 개선 사항 + 예시 화면
### step 01. 사용자 ID, 추천 타겟 연도/월 입력시 사용자 기본 정보와 과거 이력과 함께 영화 추천
![](https://velog.velcdn.com/images/qazsxdc/post/42ee0f0b-a395-4b9d-b494-28980db2997a/image.png)

### step 02. 추가적으로 의도대로 움직이는지 확인하기 위해 streamlit 구성 변경.
![](https://velog.velcdn.com/images/qazsxdc/post/6a16e150-8a52-4a77-bd28-3a74c5317c79/image.png)
![](https://velog.velcdn.com/images/qazsxdc/post/56e157b7-0a5f-47f3-b5f4-0a6a83b0459c/image.png)

1. 모델이 정상 동작하는 근거
score(모델 예측 확률)와 score_adj(최근작 가중 반영 후 점수)가 모두 다르게 계산되어 있습니다.
→ 즉, AutoInt 모델의 추론 결과 + 최근작 보정이 잘 적용됨
→ 사용자가 과거에 많이 본 장르와 연대(1980년대 드라마 등)를 기준으로 설명이 붙은 것.

2. 추천 품질 해석
- 상위권(예: Chinatown, Yojimbo, Treasure of the Sierra Madre, For All Mankind, Casablanca 등)은
전통적 명작이면서 Drama / Adventure / Mystery / Documentary 등 사용자의 주된 취향과 일치합니다.
- score_adj가 score보다 약간 높은 항목(예: Running Free, For All Mankind)은 최근 연도(2000년대 이후) 가중치(alpha=0.15)가 추가된 결과입니다.
- 장르 다양성(λ=0.10)도 작용해서 순위에 코미디, 필름누아르, 다큐멘터리 등 다양한 장르가 섞여 있음 → 다양성 로직 정상 작동.

3. 정량 평가 대시보드: AUC, Precision@K(사용자별 샘플 평가) 계산 + 시각화
- 평균 AUC = 0.7746 
  - 전체 사용자 샘플에서 모델이 ‘좋은 영화 vs 나쁜 영화’를 구분하는 능력
  - 0.5가 랜덤 / 1.0이 완벽 → 0.77은 꽤 양호한 수준 (좋은 분류 성능)
- 평균 Precision@10 = 0.9000 
  - 상위 10개 추천 중 실제로 사용자가 좋아할 영화(평점 4점 이상)의 비율
  - 10개 중 9개가 사용자의 취향에 맞았다는 뜻 → 매우 우수함
- 히스토그램 (AUC, Precision@K)
  - 사용자별 분포
  - 대부분의 사용자에서 AUC>0.7, Precision@K>0.8 이상 → 모델이 안정적
- 하단 테이블
  - 개별 사용자 성능

  - 사용자별 Precision@K과 AUC 편차를 확인 가능 → 일부 낮은 유저는 Cold Start 가능성 있음


### step 03. 정확도 상승과 배포 안정성을 목표로 모델 업데이트

| 번호    | 변경 내용 요약                                                                                               | 분류                      | 이유                                                         |
| ----- | ------------------------------------------------------------------------------------------------------ | ----------------------- | ---------------------------------------------------------- |
| **①** | 임베딩 크기·어텐션 레이어·헤드 수·DNN 깊이 확장 (기존:embed_dim=16, att_layer_num=3, att_head_num=2, dnn_hidden_units=[64, 32], dnn_dropout=0.4 → 수정:embed_dim=32, att_layer_num=4, att_head_num=4, dnn_hidden_units=[128, 64, 32], dnn_dropout=0.2) | **정확도 상승**        | 모델 용량과 표현력을 높여 복잡한 상호작용을 더 잘 학습하게 함.  |
| **②** | 추천 재랭킹 기본값(λ, α 등) 완화 → 과도한 장르 패널티 방지      |  **정확도 상승**           | 추천 다양성 제어를 완화해 실제 사용자의 선호 패턴을 더 자연스럽게 반영   |
| **③** | 가중치 로딩 방식을 `by_name=True, skip_mismatch=True`로 수정하고, `.h5` → `.weights.h5` 두 포맷 모두 지원    | **배포 안정성 강화**        | 서로 다른 Keras/TensorFlow 버전에서도 깨지지 않고 로드 가능하게 함. (환경 불일치 대비)  |
| **④** | 모델 초기화(워밍업) 유지 + 로드 실패 시 경고 메시지 명시                                                               |  **배포 안정성 강화**     | Streamlit 환경에서 로드 실패 시 앱이 멈추지 않고 사용자에게 명확한 피드백 제공              |

![](https://velog.velcdn.com/images/qazsxdc/post/622d5c78-2365-4447-aa76-29d72145c276/image.png)








