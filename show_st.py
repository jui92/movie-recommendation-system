import os, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
from autoint import AutoIntModel, predict_model

def _pick(*cands):
    for p in cands:
        p = Path(p)
        if p.exists(): return p
    return None

def _find_roots():
    here = Path(__file__).resolve().parent
    cwd = Path(os.getcwd()).resolve()
    data_dir  = _pick(cwd/"data", here/"data", here.parent/"data")
    model_dir = _pick(cwd/"model", here/"model", here.parent/"model")
    return data_dir, model_dir, here, cwd

@st.cache_resource
def load_data():
    data_dir, model_dir, here, cwd = _find_roots()
    with st.sidebar:
        st.write("**Path debug**")
        st.write("script:", str(here)); st.write("cwd:", str(cwd))
        st.write("data_dir:", str(data_dir)); st.write("model_dir:", str(model_dir))
    if data_dir is None or model_dir is None:
        raise FileNotFoundError("data/ 또는 model/ 폴더를 찾지 못했습니다.")

    field_dims = np.load(data_dir/'field_dims.npy')
    ratings_df = pd.read_csv(data_dir/'ml-1m'/'ratings_prepro.csv')
    movies_df  = pd.read_csv(data_dir/'ml-1m'/'movies_prepro.csv')
    users_df   = pd.read_csv(data_dir/'ml-1m'/'users_prepro.csv')

    model = AutoIntModel(field_dims, 16, att_layer_num=3, att_head_num=2, att_res=True,
                         dnn_hidden_units=[64,32], dnn_dropout=0.4)
    _ = model(np.zeros((1, len(field_dims)), dtype="int64"))
    model.load_weights(model_dir/'autoInt_model.weights.h5')

    encs = joblib.load(data_dir/'label_encoders.pkl')
    return users_df, movies_df, ratings_df, model, encs

def get_user_seen_movies(ratings_df):
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seed_dict(movies_df, users_df, user_seen):
    u_movies = movies_df['movie_id'].unique()
    u_users  = users_df['user_id'].unique()
    seen_map = dict(zip(user_seen['user_id'], user_seen['movie_id']))
    out = {}
    for u in u_users:
        seen = set(seen_map.get(u, []))
        out[u] = list(set(u_movies) - seen)
    return out

def _encode_with_label_encoders(df, encs):
    for col, le in encs.items():
        if col not in df.columns: continue
        vals = df[col].fillna('no')
        # 인코더 클래스 dtype에 맞춤
        if getattr(le, 'classes_', None) is not None and le.classes_.dtype.kind not in {'U','S','O'}:
            vals = pd.to_numeric(vals, errors='coerce').fillna(0).astype(int).astype(str)
        else:
            vals = vals.astype(str)
        vals = vals.where(vals.isin(le.classes_), le.classes_[0])
        df[col] = le.transform(vals)
    return df

def get_user_info(uid, users_df): return users_df[users_df['user_id']==uid]
def get_user_past(uid, ratings_df, movies_df):
    return ratings_df[(ratings_df['user_id']==uid)&(ratings_df['rating']>=4)].merge(movies_df, on='movie_id')

def get_recom(uid, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, encs, topk=20):
    cand = user_non_seen_dict.get(uid, [])
    if not cand: return pd.DataFrame(columns=list(movies_df.columns)+['score'])
    df_movies = pd.DataFrame({'movie_id': cand}).merge(movies_df, on='movie_id', how='left')
    df_user   = pd.DataFrame({'user_id':[uid]*len(df_movies)}).merge(users_df, on='user_id', how='left')
    df_user['rating_year']=r_year; df_user['rating_month']=r_month
    df_user['rating_decade']=str(r_year - (r_year%10))+'s'
    feat = ['user_id','movie_id','movie_decade','movie_year','rating_year','rating_month',
            'rating_decade','genre1','genre2','genre3','gender','age','occupation','zip']
    merge_data = pd.concat([df_movies.reset_index(drop=True), df_user.reset_index(drop=True)], axis=1)
    merge_data = merge_data[[c for c in feat if c in merge_data.columns]].fillna('no')
    merge_data = _encode_with_label_encoders(merge_data, encs)

    top = predict_model(model, merge_data, topk=topk)   # [(enc_mid, score), ...]
    if not top: return pd.DataFrame(columns=list(movies_df.columns)+['score'])
    enc_ids = np.array([t[0] for t in top])
    scores  = {int(t[0]): float(t[1]) for t in top}

    # movie_id 역변환 & dtype 정렬
    if 'movie_id' in encs:
        le = encs['movie_id']
        inv_ids = le.inverse_transform(enc_ids)
        if le.classes_.dtype.kind in {'U','S','O'}:
            movies_key = movies_df['movie_id'].astype(str)
        else:
            movies_key = pd.to_numeric(movies_df['movie_id'], errors='coerce').fillna(-1).astype(int)
    else:
        inv_ids = enc_ids
        movies_key = movies_df['movie_id']

    mask = movies_key.isin(pd.Series(inv_ids))
    out = movies_df.loc[mask].copy()

    if 'movie_id' in encs:
        le = encs['movie_id']
        if le.classes_.dtype.kind in {'U','S','O'}:
            key_for_score = out['movie_id'].astype(str)
        else:
            key_for_score = pd.to_numeric(out['movie_id'], errors='coerce').fillna(-1).astype(int)
        out_enc = le.transform(key_for_score)
        out['score'] = [scores.get(int(e), np.nan) for e in out_enc]
    else:
        out['score'] = np.nan
    return out.sort_values('score', ascending=False)

# ===== App =====
users_df, movies_df, ratings_df, model, encs = load_data()
user_seen = get_user_seen_movies(ratings_df)
user_non_seen = get_user_non_seed_dict(movies_df, users_df, user_seen)

st.title("영화 추천 결과 살펴보기 (AutoInt)")
st.header("사용자 정보를 넣어주세요.")
uid = st.number_input("사용자 ID 입력",
                      min_value=int(users_df['user_id'].min()),
                      max_value=int(users_df['user_id'].max()),
                      value=int(users_df['user_id'].min()))
ry = st.number_input("추천 타겟 연도 입력",
                     min_value=int(ratings_df['rating_year'].min()),
                     max_value=int(ratings_df['rating_year'].max()),
                     value=int(ratings_df['rating_year'].min()))
rm = st.number_input("추천 타겟 월 입력",
                     min_value=int(ratings_df['rating_month'].min()),
                     max_value=int(ratings_df['rating_month'].max()),
                     value=int(ratings_df['rating_month'].min()))

if st.button("추천 결과 보기"):
    st.subheader("사용자 기본 정보")
    st.dataframe(get_user_info(uid, users_df))

    st.subheader("사용자가 과거에 선호(평점 4점 이상)한 영화")
    st.dataframe(get_user_past(uid, ratings_df, movies_df))

    st.subheader("추천 결과 (상위 20개)")
    rec = get_recom(uid, user_non_seen, users_df, movies_df, ry, rm, model, encs, topk=20)
    st.dataframe(rec[['movie_id','title','movie_year','movie_decade','genre1','genre2','genre3','score']])
