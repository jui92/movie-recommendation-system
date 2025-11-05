import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

from autoint import AutoIntModel, predict_model

# streamlit run show_st.py

@st.cache_resource
def load_data():
    """
    Load preprocessed MovieLens 1M data & the trained AutoInt weights.

    Expected structure (relative to this file):
    ./data/
        field_dims.npy
        label_encoders.pkl
        ./ml-1m/
            movies_prepro.csv
            ratings_prepro.csv
            users_prepro.csv
    ./model/
        autoInt_model_weights.h5
    """
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"

    # Essential artifacts
    field_dims = np.load(f'{data_path}/field_dims.npy')
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    users_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')

    # Build model skeleton, then load weights
    dropout = 0.4
    embed_dim = 16
    model = AutoIntModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
                         l2_reg_dnn=0.0, l2_reg_embedding=1e-5, dnn_use_bn=False,
                         dnn_dropout=dropout, init_std=0.0001)
    # Build by calling once
    _ = model(np.zeros((1, len(field_dims)), dtype="int64"))
    model.load_weights(f'{model_path}/autoInt_model_weights.h5')

    # Label encoders (dict: column -> LabelEncoder)
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')

    return users_df, movies_df, ratings_df, model, label_encoders


def get_user_seen_movies(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """List of movies each user has seen (any rating)."""
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()


def get_user_non_seed_dict(movies_df: pd.DataFrame, users_df: pd.DataFrame, user_seen_movies: pd.DataFrame) -> dict:
    """For each user, enumerate unseen movie ids (encoded)."""
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


def get_user_past_interactions(user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Only keep movies with rating >= 4 as 'liked' examples for display."""
    return ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')


def _encode_with_label_encoders(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Apply pre-fitted encoders (transform, NOT fit) to categorical columns in-place."""
    for col, le in label_encoders.items():
        if col not in df.columns:
            continue
        # Handle unseen labels by mapping to 'unknown' if the encoder supports it; otherwise drop/clip
        try:
            df[col] = le.transform(df[col])
        except Exception:
            # Fallback: map unknowns to 0 if necessary
            known = set(getattr(le, 'classes_', []))
            df[col] = df[col].apply(lambda x: x if x in known else list(known)[0] if known else x)
            df[col] = le.transform(df[col])
    return df


def get_recom(user_id: int, user_non_seen_dict: dict, users_df: pd.DataFrame, movies_df: pd.DataFrame,
              r_year: int, r_month: int, model: AutoIntModel, label_encoders: dict, topk: int = 20) -> pd.DataFrame:
    """
    Steps:
      1) Collect user's unseen movies.
      2) Build an inference dataframe with the same columns order as training.
      3) Apply label encoders.
      4) Predict and pick top-k.
    """
    user_non_seen_movie = user_non_seen_dict.get(user_id, [])
    if not user_non_seen_movie:
        return pd.DataFrame(columns=movies_df.columns)

    user_id_list = [user_id for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'

    # Merge – create all model features
    user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id': user_non_seen_movie}), movies_df, on='movie_id', how='left')
    user_info = pd.merge(pd.DataFrame({'user_id': user_id_list}), users_df, on='user_id', how='left')
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade

    merge_data = pd.concat([user_non_seen_movie.reset_index(drop=True), user_info.reset_index(drop=True)], axis=1)
    merge_data.fillna('no', inplace=True)

    # Ensure column order matches the training pipeline (adjust if yours differs)
    feature_cols = ['user_id', 'movie_id', 'movie_decade', 'movie_year', 'rating_year',
                    'rating_month', 'rating_decade', 'genre1', 'genre2', 'genre3',
                    'gender', 'age', 'occupation', 'zip']

    # Keep only the columns we expect; silently drop extras
    merge_data = merge_data[[c for c in feature_cols if c in merge_data.columns]]

    # Apply label encoders (transform only)
    merge_data = _encode_with_label_encoders(merge_data, label_encoders)

    # Predict & pick top-k movie_ids
    top = predict_model(model, merge_data, topk=topk)  # [(encoded_movie_id, score)]
    encoded_top_ids = [t[0] for t in top]
    scores = {t[0]: t[1] for t in top}

    # Map back to original movie ids using the inverse_transform of the encoder
    if 'movie_id' in label_encoders:
        origin_m_id = label_encoders['movie_id'].inverse_transform(np.array(encoded_top_ids))
    else:
        origin_m_id = np.array(encoded_top_ids)

    # Return movie metadata for display, with a score column
    out = movies_df[movies_df['movie_id'].isin(origin_m_id)].copy()
    # create score by re-encoding movie_id to match score dict
    if 'movie_id' in label_encoders:
        re_enc = label_encoders['movie_id'].transform(out['movie_id'])
        out['score'] = [scores.get(int(e), np.nan) for e in re_enc]
    else:
        out['score'] = np.nan
    out = out.sort_values('score', ascending=False)
    return out


# ===== App =====

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
