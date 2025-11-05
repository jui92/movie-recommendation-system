사용자 ID, 추천 타겟 연도/월 입력시 사용자 기본 정보와 과거 이력과 함께 영화 추천

![](https://velog.velcdn.com/images/qazsxdc/post/42ee0f0b-a395-4b9d-b494-28980db2997a/image.png)


추가적으로 의도대로 움직이는지 확인하기 위해 streamlit 구성 변경.

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
