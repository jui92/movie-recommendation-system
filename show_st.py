import os, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
from autoint import AutoIntModel, predict_model

st.set_page_config(page_title="AutoInt Movie Recs", layout="wide")

# ---------- 경로 찾기 & 로드 ----------
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
        st.write("script:", str(here))
        st.write("cwd:", str(cwd))
        st.write("data_dir:", str(data_dir))
        st.write("model_dir:", str(model_dir))
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

# ---------- 유틸 ----------
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

# ---------- 추천 본체 ----------
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

    top = predict_model(model, merge_data, topk=max(topk, 20))   # 넉넉히 뽑아와 후처리
    if not top: return pd.DataFrame(columns=list(movies_df.columns)+['score'])
    enc_ids = np.array([t[0] for t in top])
    scores  = {int(t[0]): float(t[1]) for t in top}

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

# ---------- 재랭킹(최근작 가중) & 다양성 ----------
def rerank_by_recency(df, alpha=0.15, recent_year=2010):
    if df.empty: return df
    y = pd.to_numeric(df['movie_year'], errors='coerce').fillna(0)
    bonus = (y >= recent_year).astype(float) * float(alpha)
    df = df.copy()
    df['score_adj'] = df['score'] + bonus
    return df.sort_values('score_adj', ascending=False)

def diversify_by_genre(df, topk=20, lam=0.15, genre_cols=('genre1','genre2','genre3')):
    if df.empty: return df
    selected, seen_genres = [], set()
    pool = df.copy()
    for _ in range(min(topk, len(pool))):
        pool = pool.copy()
        def penalty(row):
            g = {row.get(c) for c in genre_cols}
            return -float(lam) * len(seen_genres & g)
        pool['div_score'] = pool.get('score_adj', pool['score']) + pool.apply(penalty, axis=1)
        pick = pool.sort_values('div_score', ascending=False).iloc[0]
        selected.append(pick)
        seen_genres |= {pick.get(c) for c in genre_cols}
        pool = pool.drop(pick.name)
    return pd.DataFrame(selected)

# ---------- 추천 이유 ----------
def reasons_for(user_like_df, rec_df):
    if rec_df.empty: return rec_df
    liked_genres = set(user_like_df[['genre1','genre2','genre3']].values.ravel()) - {None, 'None', 'no', np.nan}
    liked_decades = set(user_like_df['movie_decade'].dropna().astype(str).values)
    outs = []
    for _, r in rec_df.iterrows():
        g = [x for x in [r.get('genre1'), r.get('genre2'), r.get('genre3')] if x in liked_genres]
        d = (str(r.get('movie_decade')) in liked_decades)
        msg = []
        if g: msg.append(f"선호 장르({', '.join(g)})")
        if d: msg.append(f"선호 연대({r.get('movie_decade')})")
        outs.append("; ".join(msg) if msg else "")
    rec_df = rec_df.copy()
    rec_df['why'] = outs
    return rec_df

# ================== 앱 ==================
users_df, movies_df, ratings_df, model, encs = load_data()
user_seen = get_user_seen_movies(ratings_df)
user_non_seen = get_user_non_seed_dict(movies_df, users_df, user_seen)

st.title("영화 추천 결과 살펴보기 (AutoInt)")
st.header("사용자 정보를 넣어주세요.")

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    uid = st.number_input("사용자 ID 입력",
                          min_value=int(users_df['user_id'].min()),
                          max_value=int(users_df['user_id'].max()),
                          value=int(users_df['user_id'].min()))
with c2:
    ry = st.number_input("추천 타겟 연도 입력",
                         min_value=int(ratings_df['rating_year'].min()),
                         max_value=int(ratings_df['rating_year'].max()),
                         value=int(ratings_df['rating_year'].min()))
with c3:
    rm = st.number_input("추천 타겟 월 입력",
                         min_value=int(ratings_df['rating_month'].min()),
                         max_value=int(ratings_df['rating_month'].max()),
                         value=int(ratings_df['rating_month'].min()))
with c4:
    topk = st.slider("Top-K", min_value=5, max_value=50, value=20, step=1)

st.subheader("재랭킹 옵션")
cc1, cc2, cc3 = st.columns([1,1,1])
with cc1:
    alpha = st.slider("최근작 가중치 α", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
with cc2:
    recent_year = st.number_input("최근 기준 연도", min_value=1950, max_value=2025, value=2000, step=1)
with cc3:
    lam = st.slider("장르 다양성 λ", min_value=0.0, max_value=0.5, value=0.10, step=0.01)

if st.button("추천 결과 보기"):
    st.subheader("사용자 기본 정보")
    st.dataframe(users_df[users_df['user_id']==uid])

    st.subheader("사용자가 과거에 선호(평점 4점 이상)한 영화")
    liked = get_user_past(uid, ratings_df, movies_df)
    st.dataframe(liked)

    # 추천
    rec_raw = get_recom(uid, user_non_seen, users_df, movies_df, ry, rm, model, encs, topk=max(200, topk))
    # 중복(이미 본 영화) 체크
    seen_ids = set(ratings_df.loc[ratings_df['user_id']==uid, 'movie_id'])
    dup_cnt = rec_raw['movie_id'].isin(seen_ids).sum()
    st.caption(f"중복(이미 본 영화) 포함 개수: {dup_cnt}")

    # 재랭킹 & 다양성
    rec = rerank_by_recency(rec_raw, alpha=alpha, recent_year=int(recent_year))
    rec = diversify_by_genre(rec, topk=int(topk), lam=lam)
    rec = reasons_for(liked, rec)

    st.subheader("추천 결과 (상위 K)")
    st.dataframe(rec[['title','movie_year','movie_decade','genre1','genre2','genre3',
                      'score','score_adj','why']])
