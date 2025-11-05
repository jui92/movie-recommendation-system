import os, time, requests, joblib, numpy as np, pandas as pd, streamlit as st, altair as alt
from pathlib import Path
from autoint import AutoIntModel, predict_model

st.set_page_config(page_title="AutoInt Movie Recs (Pro)", layout="wide")

# =========================
# 경로 찾기 & 아티팩트 로딩
# =========================
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

users_df, movies_df, ratings_df, model, encs = load_data()

# =========================
# 유틸 & 인코딩
# =========================
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

# =========================
# 추천 본체
# =========================
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

    top = predict_model(model, merge_data, topk=max(topk, 50))   # 넉넉히
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

# =========================
# 재랭킹(최근작) & 다양성
# =========================
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

# =========================
# 추천 이유
# =========================
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

# =========================
# 평가 (AUC & Precision@K)
# =========================
@st.cache_data(show_spinner=False)
def evaluate_model_sample(users_sample, k=10, max_items_per_user=300):
    FEATURE_COLS = ['user_id','movie_id','movie_decade','movie_year','rating_year',
                    'rating_month','rating_decade','genre1','genre2','genre3',
                    'gender','age','occupation','zip']
    auc_list, p_at_k_list = [], []
    rows = []
    for u in users_sample:
        df_u = ratings_df[ratings_df['user_id']==u]
        if df_u.empty: continue
        # 평가용: 이 사용자가 실제로 평점 남긴 영화만으로 예측 → AUC, P@K
        tmp = df_u.merge(movies_df, on='movie_id', how='left').merge(users_df, on='user_id', how='left')
        tmp = tmp.head(max_items_per_user).copy()
        # 필요한 파생
        if 'movie_decade' not in tmp:
            tmp['movie_decade'] = (tmp.get('movie_year', pd.Series([2000]*len(tmp))) // 10 * 10).astype(str) + 's'
        if 'rating_year' not in tmp:  tmp['rating_year']  = 2000
        if 'rating_month' not in tmp: tmp['rating_month'] = 1
        if 'rating_decade' not in tmp:
            tmp['rating_decade'] = (pd.to_numeric(tmp['rating_year']) // 10 * 10).astype(str) + 's'
        for g in ['genre1','genre2','genre3']:
            if g not in tmp: tmp[g] = 'no'
        for c in ['gender','age','occupation','zip']:
            if c not in tmp: tmp[c] = 'unknown'
        tmp = tmp.fillna('no')
        # 인코딩
        enc_in = tmp[FEATURE_COLS].copy()
        enc_in = _encode_with_label_encoders(enc_in, encs)
        # 예측
        X = enc_in.values.astype('int64')
        s = model(X, training=False).numpy().reshape(-1)
        y_true = (df_u.loc[enc_in.index, 'rating'] >= 4).astype(int).values

        # AUC (유효한 경우에 한해)
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, s)
                auc_list.append(auc)
        except Exception:
            pass

        # Precision@K
        order = np.argsort(-s)[:k]
        p_at_k = y_true[order].mean() if len(order)>0 else np.nan
        p_at_k_list.append(p_at_k)

        rows.append({'user_id': u, 'auc': auc_list[-1] if len(auc_list)>0 else np.nan,
                     'precision_at_k': p_at_k})
    eval_df = pd.DataFrame(rows)
    return eval_df, float(np.nanmean(eval_df['auc'])), float(np.nanmean(eval_df['precision_at_k']))

# =========================
# 포스터/줄거리 (OMDb/TMDB)
# =========================
OMDB_KEY = st.secrets.get("OMDB_API_KEY") or os.getenv("OMDB_API_KEY")
TMDB_KEY = st.secrets.get("TMDB_API_KEY") or os.getenv("TMDB_API_KEY")

@st.cache_data(show_spinner=False)
def fetch_poster(title:str, year:int|str=None):
    # OMDb 우선, 실패 시 TMDB 간단 플로우
    if OMDB_KEY:
        try:
            params = {"t": title, "y": str(year) if year else "", "apikey": OMDB_KEY, "plot": "short"}
            r = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
            if r.ok:
                data = r.json()
                if data.get("Response") == "True":
                    poster = data.get("Poster", "")
                    plot   = data.get("Plot", "")
                    return poster if poster and poster != "N/A" else None, plot if plot and plot!="N/A" else ""
        except Exception:
            pass
    if TMDB_KEY:
        try:
            # 1) search
            r = requests.get("https://api.themoviedb.org/3/search/movie",
                             params={"api_key": TMDB_KEY, "query": title, "year": year or ""},
                             timeout=8)
            if r.ok and r.json().get("results"):
                item = r.json()["results"][0]
                poster_path = item.get("poster_path")
                overview    = item.get("overview","")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}", overview
                return None, overview
        except Exception:
            pass
    return None, ""

def show_posters(rec_df, max_cards=8):
    if rec_df.empty: 
        st.info("표시할 추천이 없습니다.")
        return
    n = min(max_cards, len(rec_df))
    cols = st.columns(4)
    for i, (_, r) in enumerate(rec_df.head(n).iterrows()):
        with cols[i % 4]:
            poster, overview = fetch_poster(str(r['title']), int(r.get('movie_year', 0)) if pd.notna(r.get('movie_year')) else None)
            st.markdown(f"**{r['title']}** ({int(r['movie_year']) if pd.notna(r['movie_year']) else 'N/A'})")
            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.caption("포스터 없음")
            if overview:
                st.caption(overview[:200] + ("..." if len(overview)>200 else ""))

# =========================
# UI 구성
# =========================
user_seen = get_user_seen_movies(ratings_df)
user_non_seen = get_user_non_seed_dict(movies_df, users_df, user_seen)

st.title("영화 추천 결과 살펴보기 (AutoInt • Pro)")

tabs = st.tabs(["추천", "평가 대시보드 (AUC/Precision@K)", "사용자 비교 (장르·연대)", "설정/도움말"])

# ---------- 탭 1: 추천 ----------
with tabs[0]:
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
        st.dataframe(get_user_info(uid, users_df))

        st.subheader("사용자가 과거에 선호(평점 4점 이상)한 영화")
        liked = get_user_past(uid, ratings_df, movies_df)
        st.dataframe(liked)

        rec_raw = get_recom(uid, user_non_seen, users_df, movies_df, ry, rm, model, encs, topk=max(200, topk))
        seen_ids = set(ratings_df.loc[ratings_df['user_id']==uid, 'movie_id'])
        dup_cnt = rec_raw['movie_id'].isin(seen_ids).sum()
        st.caption(f"중복(이미 본 영화) 포함 개수: {dup_cnt}")

        rec = rerank_by_recency(rec_raw, alpha=alpha, recent_year=int(recent_year))
        rec = diversify_by_genre(rec, topk=int(topk), lam=lam)
        rec = reasons_for(liked, rec)

        st.subheader("추천 결과 (상위 K)")
        st.dataframe(rec[['title','movie_year','movie_decade','genre1','genre2','genre3',
                          'score','score_adj','why']])

        # 포스터 카드
        show_posters_opt = st.checkbox("포스터/줄거리 카드 보기 (OMDb/TMDB 키 필요)", value=False)
        if show_posters_opt:
            show_posters(rec, max_cards=8)

# ---------- 탭 2: 평가 대시보드 ----------
with tabs[1]:
    st.header("모델 정확도: AUC / Precision@K (사용자 샘플 평가)")
    k = st.slider("K (Precision@K)", 5, 50, 10, 1)
    n_users = st.slider("평가 사용자 수 (샘플)", 20, 300, 80, 10)
    st.caption("주의: 전 사용자 전체 평가 시 비용이 큽니다. 샘플 기반으로 빠르게 점검합니다.")
    if st.button("평가 실행"):
        users_sample = list(ratings_df['user_id'].drop_duplicates().sample(n=n_users, random_state=42))
        with st.spinner("평가 중..."):
            eval_df, mean_auc, mean_p = evaluate_model_sample(users_sample, k=k, max_items_per_user=300)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("평균 AUC", f"{mean_auc:.4f}")
            auc_chart = alt.Chart(eval_df.dropna()).mark_bar().encode(
                x=alt.X('auc:Q', bin=alt.Bin(maxbins=20), title='AUC'),
                y=alt.Y('count()', title='사용자 수')
            ).properties(height=250)
            st.altair_chart(auc_chart, use_container_width=True)
        with c2:
            st.metric(f"평균 Precision@{k}", f"{mean_p:.4f}")
            p_chart = alt.Chart(eval_df.dropna()).mark_bar().encode(
                x=alt.X('precision_at_k:Q', bin=alt.Bin(maxbins=20), title=f'Precision@{k}'),
                y=alt.Y('count()', title='사용자 수')
            ).properties(height=250)
            st.altair_chart(p_chart, use_container_width=True)
        st.subheader("개별 사용자 평가 결과")
        st.dataframe(eval_df)

# ---------- 탭 3: 사용자 비교 ----------
with tabs[2]:
    st.header("특정 사용자 비교: 추천 Top-K의 장르/연대 분포")
    cc1, cc2, cc3 = st.columns([1,1,1])
    with cc1:
        u1 = st.number_input("사용자 A", min_value=int(users_df['user_id'].min()),
                             max_value=int(users_df['user_id'].max()),
                             value=int(users_df['user_id'].min()))
    with cc2:
        u2 = st.number_input("사용자 B", min_value=int(users_df['user_id'].min()),
                             max_value=int(users_df['user_id'].max()),
                             value=int(users_df['user_id'].min())+1)
    with cc3:
        comp_topk = st.slider("Top-K (비교용)", 5, 50, 20, 1)

    if st.button("비교 실행"):
        def topk_rec(uid):
            r_raw = get_recom(uid, user_non_seen, users_df, movies_df, ry=2000, r_month=1, model=model, encs=encs, topk=200)
            r = rerank_by_recency(r_raw, alpha=0.15, recent_year=2000)
            r = diversify_by_genre(r, topk=comp_topk, lam=0.10)
            return r

        r1, r2 = topk_rec(u1), topk_rec(u2)

        # 장르 탑5
        def top_genres(df):
            s = pd.Series(pd.concat([df['genre1'], df['genre2'], df['genre3']])).replace({None:'None'}).value_counts()
            s = s[s.index.notna()]
            s = s[s.index!='None']
            return s.head(5).reset_index(names=['genre', 'count'])
        g1, g2 = top_genres(r1), top_genres(r2)
        g1['user']='A'; g2['user']='B'
        g = pd.concat([g1,g2], ignore_index=True)
        g_chart = alt.Chart(g).mark_bar().encode(
            x=alt.X('genre:N', sort='-y'),
            y='count:Q',
            color='user:N',
            column='user:N'
        ).resolve_scale(y='independent').properties(height=300)
        st.subheader("Top-5 추천 장르 분포")
        st.altair_chart(g_chart, use_container_width=True)

        # 연대 분포
        def decade_dist(df):
            s = df['movie_decade'].astype(str).value_counts().reset_index(names=['movie_decade','count'])
            return s
        d1, d2 = decade_dist(r1), decade_dist(r2)
        d1['user']='A'; d2['user']='B'
        d = pd.concat([d1,d2], ignore_index=True)
        d_chart = alt.Chart(d).mark_bar().encode(
            x=alt.X('movie_decade:N', sort='-y'),
            y='count:Q',
            color='user:N',
            column='user:N'
        ).resolve_scale(y='independent').properties(height=300)
        st.subheader("추천 연대(Decade) 분포")
        st.altair_chart(d_chart, use_container_width=True)

        st.subheader("A 사용자 추천 Top-K 미리보기")
        st.dataframe(r1[['title','movie_year','movie_decade','genre1','genre2','genre3','score']])
        st.subheader("B 사용자 추천 Top-K 미리보기")
        st.dataframe(r2[['title','movie_year','movie_decade','genre1','genre2','genre3','score']])

