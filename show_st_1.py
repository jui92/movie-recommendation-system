
# show_st.py (streamlined: 추천 + 평가 대시보드)
import os, joblib, numpy as np, pandas as pd, streamlit as st, altair as alt
from pathlib import Path
from autoint import AutoIntModel, predict_model

st.set_page_config(page_title="AutoInt Movie Recs (Optimized)", layout="wide")

def _pick(*cands):
    for p in cands:
        p = Path(p)
        if p.exists(): return p
    return None

def _find_roots():
    here = Path(__file__).resolve().parent
    cwd  = Path(os.getcwd()).resolve()
    data_dir  = _pick(cwd/'data', here/'data', here.parent/'data')
    model_dir = _pick(cwd/'model', here/'model', here.parent/'model')
    return data_dir, model_dir, here, cwd

@st.cache_resource
def load_data():
    data_dir, model_dir, here, cwd = _find_roots()
    with st.sidebar:
        st.write('**Path debug**')
        st.write('script:', str(here))
        st.write('cwd:', str(cwd))
        st.write('data_dir:', str(data_dir))
        st.write('model_dir:', str(model_dir))
    if data_dir is None or model_dir is None:
        raise FileNotFoundError('data/ 또는 model/ 폴더를 찾을 수 없습니다.')

    field_dims = np.load(data_dir/'field_dims.npy')
    ratings_df = pd.read_csv(data_dir/'ml-1m'/'ratings_prepro.csv')
    movies_df  = pd.read_csv(data_dir/'ml-1m'/'movies_prepro.csv')
    users_df   = pd.read_csv(data_dir/'ml-1m'/'users_prepro.csv')

    model = AutoIntModel(field_dims, embed_dim=32, att_layer_num=4, att_head_num=4,
                         att_res=True, dnn_hidden_units=[128,64,32], dnn_dropout=0.2)
    _ = model(np.zeros((1, len(field_dims)), dtype='int64'))
    model.load_weights(model_dir/'autoInt_model.weights.h5')

    encs = joblib.load(data_dir/'label_encoders.pkl')
    return users_df, movies_df, ratings_df, model, encs

users_df, movies_df, ratings_df, model, encs = load_data()

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

def get_user_info(uid): return users_df[users_df['user_id']==uid]
def get_user_past(uid):
    return ratings_df[(ratings_df['user_id']==uid)&(ratings_df['rating']>=4)].merge(movies_df, on='movie_id')

def get_recom(uid, user_non_seen_dict, r_year, r_month, topk=20):
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

    top = predict_model(model, merge_data, topk=max(topk, 50))
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

def rerank_by_recency(df, alpha=0.15, recent_year=2010):
    if df.empty: return df
    y = pd.to_numeric(df['movie_year'], errors='coerce').fillna(0)
    df = df.copy()
    df['score_adj'] = df['score'] + (y >= recent_year).astype(float) * float(alpha)
    return df.sort_values('score_adj', ascending=False)

def diversify_by_genre(df, topk=20, lam=0.10):
    if df.empty: return df
    selected, seen = [], set()
    pool = df.copy()
    for _ in range(min(topk, len(pool))):
        pool = pool.copy()
        def penalty(row):
            g = {row.get('genre1'), row.get('genre2'), row.get('genre3')}
            return -float(lam) * len(seen & g)
        pool['div_score'] = pool.get('score_adj', pool['score']) + pool.apply(penalty, axis=1)
        pick = pool.sort_values('div_score', ascending=False).iloc[0]
        selected.append(pick); seen |= {pick.get('genre1'), pick.get('genre2'), pick.get('genre3')}
        pool = pool.drop(pick.name)
    return pd.DataFrame(selected)

def reasons_for(user_like_df, rec_df):
    if rec_df.empty: return rec_df
    liked_genres = set(user_like_df[['genre1','genre2','genre3']].values.ravel()) - {None, 'None', 'no', np.nan}
    liked_decades = set(user_like_df['movie_decade'].dropna().astype(str).values)
    outs = []
    for _, r in rec_df.iterrows():
        g = [x for x in [r.get('genre1'), r.get('genre2'), r.get('genre3')] if x in liked_genres]
        d = (str(r.get('movie_decade')) in liked_decades)
        msg = []
        if g: msg.append(f'선호 장르({', '.join(g)})')
        if d: msg.append(f'선호 연대({r.get('movie_decade')})')
        outs.append('; '.join(msg) if msg else '')
    rec_df = rec_df.copy()
    rec_df['why'] = outs
    return rec_df

@st.cache_data(show_spinner=False)
def evaluate_model_sample(users_sample, k=10, max_items_per_user=300):
    FEATURE_COLS = ['user_id','movie_id','movie_decade','movie_year','rating_year',
                    'rating_month','rating_decade','genre1','genre2','genre3',
                    'gender','age','occupation','zip']
    rows = []
    users_sample = list(pd.unique(pd.Series(users_sample)))
    for u in users_sample:
        df_u = ratings_df[ratings_df['user_id']==u]
        if df_u.empty: continue
        tmp = (
            df_u.merge(movies_df, on='movie_id', how='left')
                .merge(users_df, on='user_id', how='left')
                .head(max_items_per_user)
                .reset_index(drop=True)
                .copy()
        )
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

        enc_in = _encode_with_label_encoders(tmp[FEATURE_COLS].copy(), encs)
        X = enc_in.values.astype('int64')
        scores = model(X, training=False).numpy().reshape(-1)

        if 'rating' in tmp.columns:
            y_true = (pd.to_numeric(tmp['rating'], errors='coerce').fillna(0) >= 4).astype(int).values
        else:
            continue

        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan
        except Exception:
            auc = np.nan

        order = np.argsort(-scores)[:k]
        p_at_k = float(np.mean(y_true[order])) if len(order) > 0 else np.nan
        rows.append({'user_id': u, 'auc': auc, 'precision_at_k': p_at_k})

    eval_df = pd.DataFrame(rows)
    mean_auc = float(np.nanmean(eval_df['auc'])) if not eval_df.empty else np.nan
    mean_p   = float(np.nanmean(eval_df['precision_at_k'])) if not eval_df.empty else np.nan
    return eval_df, mean_auc, mean_p

user_seen = get_user_seen_movies(ratings_df)
user_non_seen = get_user_non_seed_dict(movies_df, users_df, user_seen)

tabs = st.tabs(['추천', '평가 대시보드 (AUC/Precision@K)'])

with tabs[0]:
    st.title('영화 추천 (AutoInt • Optimized)')
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        uid = st.number_input('사용자 ID',
                              min_value=int(users_df['user_id'].min()),
                              max_value=int(users_df['user_id'].max()),
                              value=int(users_df['user_id'].min()))
    with c2:
        ry = st.number_input('추천 타겟 연도',
                             min_value=int(ratings_df['rating_year'].min()),
                             max_value=int(ratings_df['rating_year'].max()),
                             value=int(ratings_df['rating_year'].min()))
    with c3:
        rm = st.number_input('추천 타겟 월',
                             min_value=int(ratings_df['rating_month'].min()),
                             max_value=int(ratings_df['rating_month'].max()),
                             value=int(ratings_df['rating_month'].min()))
    with c4:
        topk = st.slider('Top-K', 5, 50, 20, 1)

    st.subheader('재랭킹 옵션')
    cc1, cc2, cc3 = st.columns([1,1,1])
    with cc1: alpha = st.slider('최근작 가중치 α', 0.0, 0.5, 0.15, 0.01)
    with cc2: recent_year = st.number_input('최근 기준 연도', 1950, 2025, 2000, 1)
    with cc3: lam = st.slider('장르 다양성 λ', 0.0, 0.5, 0.10, 0.01)

    if st.button('추천 결과 보기'):
        st.subheader('사용자 기본 정보')
        st.dataframe(get_user_info(uid))
        st.subheader('사용자가 과거에 선호(평점 4+)한 영화')
        liked = get_user_past(uid); st.dataframe(liked)

        rec_raw = get_recom(uid, user_non_seen, r_year=ry, r_month=rm, topk=max(200, topk))
        seen_ids = set(ratings_df.loc[ratings_df['user_id']==uid, 'movie_id'])
        dup_cnt = rec_raw['movie_id'].isin(seen_ids).sum()
        st.caption(f'중복(이미 본 영화) 포함 개수: {dup_cnt}')

        rec = rerank_by_recency(rec_raw, alpha=float(alpha), recent_year=int(recent_year))
        rec = diversify_by_genre(rec, topk=int(topk), lam=float(lam))
        rec = reasons_for(liked, rec)

        st.subheader('추천 결과 (상위 K)')
        st.dataframe(rec[['title','movie_year','movie_decade','genre1','genre2','genre3','score','score_adj','why']])

with tabs[1]:
    st.title('평가 대시보드: AUC / Precision@K')
    k = st.slider('K (Precision@K)', 5, 50, 10, 1)
    n_users = st.slider('평가 사용자 수 (샘플)', 20, 300, 80, 10)
    if st.button('평가 실행'):
        users_sample = list(ratings_df['user_id'].drop_duplicates().sample(n=n_users, random_state=42))
        with st.spinner('평가 중...'):
            eval_df, mean_auc, mean_p = evaluate_model_sample(users_sample, k=k, max_items_per_user=300)

        c1, c2 = st.columns(2)
        with c1:
            st.metric('평균 AUC', f'{mean_auc:.4f}' if pd.notna(mean_auc) else 'NaN')
            if not eval_df.dropna(subset=['auc']).empty:
                auc_chart = alt.Chart(eval_df.dropna(subset=['auc'])).mark_bar().encode(
                    x=alt.X('auc:Q', bin=alt.Bin(maxbins=20), title='AUC'),
                    y=alt.Y('count()', title='사용자 수')
                ).properties(height=260)
                st.altair_chart(auc_chart, use_container_width=True)
        with c2:
            st.metric(f'평균 Precision@{k}', f'{mean_p:.4f}' if pd.notna(mean_p) else 'NaN')
            if not eval_df.dropna(subset=['precision_at_k']).empty:
                p_chart = alt.Chart(eval_df.dropna(subset=['precision_at_k'])).mark_bar().encode(
                    x=alt.X('precision_at_k:Q', bin=alt.Bin(maxbins=20), title=f'Precision@{k}'),
                    y=alt.Y('count()', title='사용자 수')
                ).properties(height=260)
                st.altair_chart(p_chart, use_container_width=True)
        st.subheader('개별 사용자 평가 결과')
        st.dataframe(eval_df)
