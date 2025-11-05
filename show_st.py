import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

from autoint import AutoIntModel, predict_model

@st.cache_resource
def load_data():
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"

    field_dims = np.load(f'{data_path}/field_dims.npy')
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    users_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')

    model = AutoIntModel(field_dims, 16, att_layer_num=3, att_head_num=2, att_res=True,
                         l2_reg_dnn=0.0, l2_reg_embedding=1e-5, dnn_use_bn=False,
                         dnn_dropout=0.4, init_std=0.0001)
    _ = model(np.zeros((1, len(field_dims)), dtype="int64"))
    model.load_weights(f'{model_path}/autoInt_model.weights.h5')

    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    return users_df, movies_df, ratings_df, model, label_encoders

def get_user_seen_movies(ratings_df: pd.DataFrame) -> pd.DataFrame:
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seed_dict(movies_df, users_df, user_seen_movies):
    unique_movies = movies_df['movie_id'].unique()
    unique_users = users_df['user_id'].unique()
    user_non_seen_dict = {}
    seen_map = dict(zip(user_seen_movies['user_id'], user_seen_movies['movie_id']))
    for user in unique_users:
        user_seen_list = seen_map.get(user, [])
        user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_list))
        user_non_seen_dict[user] = user_non_seen_movie_list
    return user_non_seen_dict

def get_user_info(user_id: int, users_df: pd.DataFrame) -> pd.DataFrame:
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id, ratings_df, movies_df):
    return ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')

def _encode_with_label_encoders(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    for col, le in label_encoders.items():
        if col not in df.columns:
            continue
        try:
            df[col] = le.transform(df[col])
        except Exception:
            known = list(getattr(le, 'classes_', []))
            if known:
                mapper = {v: v for v in known}
                df[col] = df[col].map(lambda x: x if x in mapper else known[0])
                df[col] = le.transform(df[col])
            else:
                df[col] = 0
    return df

def get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders, topk=20):
    user_non_seen_movie = user_non_seen_dict.get(user_id, [])
    if not user_non_seen_movie:
        return pd.DataFrame(columns=movies_df.columns)

    user_id_list = [user_id for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'

    user_non_seen_movie = pd.merge(
        pd.DataFrame({'movie_id': user_non_seen_movie}),
        movies_df, on='movie_id', how='left'
    )
    user_info = pd.merge(
        pd.DataFrame({'user_id': user_id_list}),
        users_df, on='user_id', how='left'
    )
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade

    feature_cols = ['user_id', 'movie_id', 'movie_decade', 'movie_year', 'rating_year',
                    'rating_month', 'rating_decade', 'genre1', 'genre2', 'genre3',
                    'gender', 'age', 'occupation', 'zip']

    merge_data = pd.concat(
        [user_non_seen_movie.reset_index(drop=True),
         user_info.reset_index(drop=True)],
        axis=1
    )

    merge_data = merge_data[[c for c in feature_cols if c in merge_data.columns]]
    merge_data = merge_data.fillna('no')

    # === 인코딩: 라벨인코더 classes_ 타입과 입력 타입을 일치시킴 ===
    def _safe_transform(col):
        if col in label_encoders:
            le = label_encoders[col]
            # 라벨 클래스 dtype 확인
            cls = getattr(le, 'classes_', None)
            if cls is not None:
                if cls.dtype.kind in {'U','S','O'}:   # 문자열 기반 인코더
                    merge_data[col] = merge_data[col].astype(str)
                else:                                  # 숫자 기반 인코더
                    # 숫자형으로 깔끔히
                    merge_data[col] = pd.to_numeric(merge_data[col], errors='coerce').fillna(0).astype(int)
            # 변환 시도
            try:
                merge_data[col] = le.transform(merge_data[col])
            except Exception:
                # 미지 클래스는 최빈/첫번째 클래스로 폴백
                if cls is not None and len(cls) > 0:
                    fallback = cls[0]
                    merge_data[col] = merge_data[col].map(lambda x: x if x in cls else fallback)
                    merge_data[col] = le.transform(merge_data[col])
                else:
                    merge_data[col] = 0
        return

    for c in feature_cols:
        _safe_transform(c)

    # === 예측 ===
    top = predict_model(model, merge_data, topk=topk)  # [(encoded_movie_id, score), ...]
    if len(top) == 0:
        return pd.DataFrame(columns=list(movies_df.columns) + ['score'])

    enc_movie_ids = np.array([t[0] for t in top])
    scores_map = {int(t[0]): float(t[1]) for t in top}

    # === 역변환 & 조인 dtype 정렬 ===
    if 'movie_id' in label_encoders:
        le_m = label_encoders['movie_id']
        cls = le_m.classes_
        if cls.dtype.kind in {'U','S','O'}:
            # 인코더가 문자열 클래스인 경우: 역변환 → 문자열
            origin_movie_ids = le_m.inverse_transform(enc_movie_ids)
            # movies_df key도 문자열로 맞춤
            movies_key = movies_df['movie_id'].astype(str)
        else:
            # 인코더가 숫자 클래스인 경우: 역변환 → 숫자
            origin_movie_ids = le_m.inverse_transform(enc_movie_ids)
            movies_key = pd.to_numeric(movies_df['movie_id'], errors='coerce').fillna(-1).astype(int)
    else:
        # 인코더 없음: 인코딩된 값 자체가 원본과 같을 가능성 낮음 → 안전하게 공집합 방지 로깅
        origin_movie_ids = enc_movie_ids
        movies_key = movies_df['movie_id']

    # 필터링
    mask = movies_key.isin(pd.Series(origin_movie_ids))
    out = movies_df.loc[mask].copy()

    # 점수 매핑: out의 movie_id도 인코딩해서 점수 붙임 (dtype 맞춘 뒤 변환)
    if 'movie_id' in label_encoders:
        le_m = label_encoders['movie_id']
        # out 키를 인코더 입력 dtype에 맞춰 변환
        if le_m.classes_.dtype.kind in {'U','S','O'}:
            out_key_for_score = out['movie_id'].astype(str)
        else:
            out_key_for_score = pd.to_numeric(out['movie_id'], errors='coerce').fillna(-1).astype(int)
        try:
            out_enc = le_m.transform(out_key_for_score)
        except Exception:
            # 역변환이 잘 되었으면 여기서도 보통 안전하지만, 혹시 모를 에러에 대비
            out['score'] = np.nan
        else:
            out['score'] = [scores_map.get(int(e), np.nan) for e in out_enc]
    else:
        out['score'] = np.nan

    # 최종 정렬
    out = out.sort_values('score', ascending=False)
    return out


users_df, movies_df, ratings_df, model, label_encoders = load_data()
user_seen_movies = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

st.title("영화 추천 결과 살펴보기 (AutoInt)")
st.header("사용자 정보를 넣어주세요.")
user_id = st.number_input("사용자 ID 입력", min_value=int(users_df['user_id'].min()), max_value=int(users_df['user_id'].max()), value=int(users_df['user_id'].min()))
r_year = st.number_input("추천 타겟 연도 입력", min_value=int(ratings_df['rating_year'].min()), max_value=int(ratings_df['rating_year'].max()), value=int(ratings_df['rating_year'].min()))
r_month = st.number_input("추천 타겟 월 입력", min_value=int(ratings_df['rating_month'].min()), max_value=int(ratings_df['rating_month'].max()), value=int(ratings_df['rating_month'].min()))

if st.button("추천 결과 보기"):
    st.subheader("사용자 기본 정보")
    st.dataframe(get_user_info(user_id, users_df))

    st.subheader("사용자가 과거에 선호(평점 4점 이상)한 영화")
    st.dataframe(get_user_past_interactions(user_id, ratings_df, movies_df))

    st.subheader("추천 결과 (상위 20개)")
    recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders, topk=20)
    st.dataframe(recommendations)
