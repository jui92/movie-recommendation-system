
# show_st.py (compatible with dict encoders or sklearn LabelEncoder)
import os, joblib, numpy as np, pandas as pd, streamlit as st, altair as alt
from pathlib import Path
from autoint import AutoIntModel, predict_model
import numpy as np

st.set_page_config(page_title='AutoInt Movie Recs (Improved)', layout='wide')

FEATURE_COLS = ['user_id','movie_id','movie_decade','movie_year','rating_year',
                'rating_month','rating_decade','genre1','genre2','genre3',
                'gender','age','occupation','zip']

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

def _is_dict_encoders(encs):
    return isinstance(encs, dict) and all(isinstance(v, dict) for v in encs.values())

def _transform_with_encoders(df, encs):
    out = df.copy()
    if _is_dict_encoders(encs):
        for c, mp in encs.items():
            if c not in out: continue
            v = out[c].astype(str).fillna('no')
            if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
                v = pd.to_numeric(v, errors='coerce').fillna(0).astype(int).astype(str)
            unk = mp.get('<<UNK>>', 0)
            out[c] = v.map(mp).fillna(unk).astype(int)
        return out

    for col, le in encs.items():
        if col not in out.columns: continue
        vals = out[col].fillna('no')
        try:
            if getattr(le, 'classes_', None) is not None and le.classes_.dtype.kind not in {'U','S','O'}:
                vals = pd.to_numeric(vals, errors='coerce').fillna(0).astype(int).astype(str)
            else:
                vals = vals.astype(str)
            vals = vals.where(vals.isin(le.classes_), le.classes_[0])
            out[col] = le.transform(vals)
        except Exception:
            try:
                mp = {c:i for i,c in enumerate(getattr(le, 'classes_', []))}
                out[col] = vals.map(mp).fillna(0).astype(int)
            except Exception:
                out[col] = 0
    return out

@st.cache_resource(show_spinner=False)
def load_data():
    data_dir, model_dir, here, cwd = _find_roots()
    with st.sidebar:
        st.write('**Path debug**')
        st.write('script:', str(here)); st.write('cwd:', str(cwd))
        st.write('data_dir:', str(data_dir)); st.write('model_dir:', str(model_dir))
    if data_dir is None or model_dir is None:
        raise FileNotFoundError('data/ 또는 model/ 을 찾을 수 없습니다.')

    field_dims = np.load(data_dir/'field_dims.npy')
    ratings_df = pd.read_csv(data_dir/'ml-1m'/'ratings_prepro.csv')
    movies_df  = pd.read_csv(data_dir/'ml-1m'/'movies_prepro.csv')
    users_df   = pd.read_csv(data_dir/'ml-1m'/'users_prepro.csv')
    encs = joblib.load(data_dir/'label_encoders.pkl')

    model = AutoIntModel(field_dims, embed_dim=48, att_layer_num=4, att_head_num=4,
                         att_res=True, dnn_hidden_units=[256,128,64], dnn_dropout=0.25)
    _ = model(np.zeros((1, len(field_dims)), dtype='int64'))

    legacy_h5 = model_dir/'autoInt_model.h5'
    keras3_w  = model_dir/'autoInt_model.weights.h5'
    loaded = False
    if legacy_h5.exists():
        try:
            model.load_weights(str(legacy_h5), by_name=True, skip_mismatch=True)
            loaded = True
        except Exception as e:
            st.warning(f'Legacy H5 로드 실패: {e}')
    if not loaded and keras3_w.exists():
        try:
            model.load_weights(str(keras3_w))
            loaded = True
        except Exception as e:
            st.warning(f'Keras3 weights 로드 실패: {e}')
    if not loaded:
        st.warning('가중치를 로드하지 못했습니다. 학습을 먼저 수행하세요 (train_autoint_plus.py).')

    return users_df, movies_df, ratings_df, model, encs, field_dims

users_df, movies_df, ratings_df, model, encs, field_dims = load_data()

def get_user_seen_movies(ratings_df):
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seed_dict(movies_df, users_df, user_seen):
    u_movies = movies_df['movie_id'].unique()
    u_users  = users_df['user_id'].unique()
    seen_map = dict(zip(user_seen['user_id'], user_seen['movie_id']))
    return {u: list(set(u_movies) - set(seen_map.get(u, []))) for u in u_users}

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
    feat = FEATURE_COLS
    merge_data = pd.concat([df_movies.reset_index(drop=True), df_user.reset_index(drop=True)], axis=1)
    merge_data = merge_data[[c for c in feat if c in merge_data.columns]].fillna('no')
    merge_data = _transform_with_encoders(merge_data, encs)

    top = predict_model(model, merge_data, topk=max(topk, 50))
    if not top: return pd.DataFrame(columns=list(movies_df.columns)+['score'])
    enc_ids = np.array([t[0] for t in top]); scores = {int(t[0]): float(t[1]) for t in top}

    if isinstance(encs.get('movie_id'), dict):
        inv_map = {v:k for k,v in encs['movie_id'].items() if k != '<<UNK>>'}
        inv_ids = np.array([inv_map.get(int(e), None) for e in enc_ids])
        movies_key = movies_df['movie_id'].astype(str)
    else:
        le = encs['movie_id']; inv_ids = le.inverse_transform(enc_ids)
        movies_key = movies_df['movie_id'].astype(str) if le.classes_.dtype.kind in {'U','S','O'} \
                     else pd.to_numeric(movies_df['movie_id'], errors='coerce').fillna(-1).astype(int)

    mask = movies_key.isin(pd.Series(inv_ids).astype(str)); out = movies_df.loc[mask].copy()

    if isinstance(encs.get('movie_id'), dict):
        key_for_score = out['movie_id'].astype(str).map(encs['movie_id']).fillna(encs['movie_id'].get('<<UNK>>', 0)).astype(int)
        out['score'] = [scores.get(int(e), np.nan) for e in key_for_score]
    else:
        le = encs['movie_id']
        key_for_score = out['movie_id'].astype(str) if le.classes_.dtype.kind in {'U','S','O'} \
                        else pd.to_numeric(out['movie_id'], errors='coerce').fillna(-1).astype(int)
        out_enc = le.transform(key_for_score); out['score'] = [scores.get(int(e), np.nan) for e in out_enc]
    return out.sort_values('score', ascending=False)

def rerank_by_recency(df, alpha=0.15, recent_year=2010):
    if df.empty: return df
    y = pd.to_numeric(df['movie_year'], errors='coerce').fillna(0)
    df = df.copy(); df['score_adj'] = df['score'] + (y >= recent_year).astype(float)*float(alpha)
    return df.sort_values('score_adj', ascending=False)

def diversify_by_genre(df, topk=20, lam=0.10):
    if df.empty: return df
    selected, seen = [], set(); pool = df.copy()
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
        if g: msg.append("선호 장르(" + ', '.join(g) + ")")
        if d: msg.append(f"선호 연대({r.get('movie_decade')})")
        outs.append('; '.join(msg) if msg else '')
    rec_df = rec_df.copy(); rec_df['why'] = outs
    return rec_df

users_df, movies_df, ratings_df = users_df, movies_df, ratings_df
user_seen = get_user_seen_movies(ratings_df); user_non_seen = get_user_non_seed_dict(movies_df, users_df, user_seen)
tabs = st.tabs(['추천', '평가 대시보드 (AUC/Precision@K/Recall@K/NDCG)'])

with tabs[0]:
    st.title('영화 추천 (AutoInt • Improved)')
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        uid = st.number_input('사용자 ID', min_value=int(users_df['user_id'].min()),
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
        st.subheader('사용자 기본 정보'); st.dataframe(get_user_info(uid))
        st.subheader('사용자가 과거에 선호(평점 4+)한 영화'); liked = get_user_past(uid); st.dataframe(liked)
        rec_raw = get_recom(uid, user_non_seen, r_year=ry, r_month=rm, topk=max(200, topk))
        rec = rerank_by_recency(rec_raw, alpha=float(alpha), recent_year=int(recent_year))
        rec = diversify_by_genre(rec, topk=int(topk), lam=float(lam))
        rec = reasons_for(liked, rec)
        show_cols = [c for c in ['title','movie_year','movie_decade','genre1','genre2','genre3','score','score_adj','why'] if c in rec.columns]
        st.subheader('추천 결과 (상위 K)')
        st.dataframe(rec[show_cols])

def _eval_one_user(u, k=10, max_items_per_user=300):
    df_u = ratings_df[ratings_df['user_id']==u]
    if df_u.empty: return None
    tmp = (df_u.merge(movies_df, on='movie_id', how='left')
               .merge(users_df, on='user_id', how='left')
               .head(max_items_per_user).reset_index(drop=True).copy())
    if 'movie_decade' not in tmp:
        tmp['movie_decade'] = (pd.to_numeric(tmp.get('movie_year', 2000), errors='coerce').fillna(2000)//10*10).astype(int).astype(str)+'s'
    if 'rating_year' not in tmp:  tmp['rating_year']  = 2000
    if 'rating_month' not in tmp: tmp['rating_month'] = 1
    if 'rating_decade' not in tmp:
        tmp['rating_decade'] = (pd.to_numeric(tmp['rating_year'], errors='coerce').fillna(2000)//10*10).astype(int).astype(str)+'s'
    tmp = tmp.fillna('no')

    enc_in = _transform_with_encoders(tmp[FEATURE_COLS].copy(), encs)
    X = enc_in.values.astype('int64'); scores = model(X, training=False).numpy().reshape(-1)
    if 'rating' not in tmp.columns: return None
    y_true = (pd.to_numeric(tmp['rating'], errors='coerce').fillna(0) >= 4).astype(int).values

    order = np.argsort(-scores)[:k]
    p_at_k = float(np.mean(y_true[order])) if len(order)>0 else np.nan
    tp = float(np.sum(y_true[order])); pos = float(np.sum(y_true))
    r_at_k = float(tp/pos) if pos>0 else np.nan
    gains = (2**y_true[order]-1)/np.log2(np.arange(2, 2+len(order)))
    dcg = float(np.sum(gains))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum((2**ideal-1)/np.log2(np.arange(2, 2+len(ideal)))))
    ndcg = float(dcg/idcg) if idcg>0 else np.nan
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true))>1 else np.nan
    except Exception:
        auc = np.nan
    return {'user_id': u, 'auc': auc, 'precision_at_k': p_at_k, 'recall_at_k': r_at_k, 'ndcg_at_k': ndcg}

with tabs[1]:
    st.title('평가 대시보드: AUC / Precision@K / Recall@K / NDCG@K')
    k = st.slider('K', 5, 50, 10, 1)
    n_users = st.slider('평가 사용자 수 (샘플)', 20, 300, 80, 10)
    if st.button('평가 실행'):
        users_sample = list(ratings_df['user_id'].drop_duplicates().sample(n=n_users, random_state=42))
        with st.spinner('평가 중...'):
            rows = []
            for u in users_sample:
                r = _eval_one_user(u, k=k, max_items_per_user=300)
                if r: rows.append(r)
            eval_df = pd.DataFrame(rows)

        if not eval_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.metric('평균 AUC', f"{np.nanmean(eval_df['auc']):.4f}")
                st.metric(f'평균 Precision@{k}', f"{np.nanmean(eval_df['precision_at_k']):.4f}")
            with c2:
                st.metric(f'평균 Recall@{k}', f"{np.nanmean(eval_df['recall_at_k']):.4f}")
                st.metric(f'평균 NDCG@{k}', f"{np.nanmean(eval_df['ndcg_at_k']):.4f}")
            st.subheader('분포')
            for col in ['auc','precision_at_k','recall_at_k','ndcg_at_k']:
                dfp = eval_df.dropna(subset=[col])
                if dfp.empty: continue
                chart = alt.Chart(dfp).mark_bar().encode(
                    x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=20), title=col),
                    y=alt.Y('count()', title='사용자 수')
                ).properties(height=220)
                st.altair_chart(chart, use_container_width=True)
            st.subheader('개별 사용자 평가 결과'); st.dataframe(eval_df)
        else:
            st.info('평가 가능한 사용자 샘플이 없습니다.')
