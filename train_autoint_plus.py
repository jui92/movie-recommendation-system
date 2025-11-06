import os, json, joblib, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.metrics import roc_auc_score
from autoint import AutoIntModel

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ML_DIR   = DATA_DIR / "ml-1m"
MODEL_DIR= BASE / "model"
for p in [DATA_DIR, ML_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ['user_id','movie_id','movie_decade','movie_year','rating_year',
                'rating_month','rating_decade','genre1','genre2','genre3',
                'gender','age','occupation','zip']

def load_csvs():
    movies  = pd.read_csv(ML_DIR/'movies_prepro.csv')
    ratings = pd.read_csv(ML_DIR/'ratings_prepro.csv')
    users   = pd.read_csv(ML_DIR/'users_prepro.csv')
    return movies, ratings, users

def ensure_features(movies, ratings, users):
    df = ratings.merge(movies, on='movie_id', how='left').merge(users, on='user_id', how='left')
    if 'movie_decade' not in df:
        df['movie_decade'] = (pd.to_numeric(df.get('movie_year', 2000), errors='coerce').fillna(2000)//10*10).astype(int).astype(str)+'s'
    if 'rating_year' not in df:  df['rating_year']  = 2000
    if 'rating_month' not in df: df['rating_month'] = 1
    if 'rating_decade' not in df:
        df['rating_decade'] = (pd.to_numeric(df['rating_year'], errors='coerce').fillna(2000)//10*10).astype(int).astype(str)+'s'
    for g in ['genre1','genre2','genre3']:
        if g not in df: df[g] = 'no'
    for c in ['gender','age','occupation','zip']:
        if c not in df: df[c] = 'unknown'
    keep = FEATURE_COLS + (['rating'] if 'rating' in df else [])
    return df[keep].copy()

def build_encoders(df):
    encs = {}
    for c in FEATURE_COLS:
        vals = df[c].astype(str).fillna('no')
        if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
            vals = pd.to_numeric(vals, errors='coerce').fillna(0).astype(int).astype(str)
        uniq = pd.Index(vals.unique().tolist() + ['<<UNK>>']).astype(str).unique()
        encs[c] = {v:i for i,v in enumerate(uniq)}
    return encs

def transform_with_dict(df, encs):
    out = df.copy()
    for c, mapping in encs.items():
        v = out[c].astype(str).fillna('no')
        if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
            v = pd.to_numeric(v, errors='coerce').fillna(0).astype(int).astype(str)
        out[c] = v.map(mapping).fillna(mapping.get('<<UNK>>', 0)).astype(int)
    return out

def precision_at_k(scores, y_true, k=10):
    order = np.argsort(-scores)[:k]
    return float(np.mean(y_true[order])) if len(order) else np.nan

def recall_at_k(scores, y_true, k=10):
    order = np.argsort(-scores)[:k]
    tp = float(np.sum(y_true[order]))
    pos = float(np.sum(y_true))
    return float(tp/pos) if pos > 0 else np.nan

def ndcg_at_k(scores, y_true, k=10):
    order = np.argsort(-scores)[:k]
    gains = (2**y_true[order]-1)/np.log2(np.arange(2, 2+len(order)))
    dcg = float(np.sum(gains))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum((2**ideal-1)/np.log2(np.arange(2, 2+len(ideal)))))
    return float(dcg/idcg) if idcg>0 else np.nan

def focal_loss(gamma=2.0, alpha=0.25):
    def _loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        w  = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -tf.reduce_mean(w * tf.pow(1. - pt, gamma) * tf.math.log(pt + eps))
    return _loss

if __name__ == "__main__":
    movies, ratings, users = load_csvs()
    df = ensure_features(movies, ratings, users)

    encs = build_encoders(df)
    joblib.dump(encs, DATA_DIR/'label_encoders.pkl')
    field_dims = np.array([len(encs[c]) for c in FEATURE_COLS], dtype='int32')
    np.save(DATA_DIR/'field_dims.npy', field_dims)
    df_enc = transform_with_dict(df, encs)

    if 'rating' not in df_enc.columns:
        raise ValueError("ratings_prepro.csv must include 'rating' column.")
    y = (pd.to_numeric(df_enc['rating'], errors='coerce').fillna(0) >= 4).astype('int32').values
    X = df_enc[FEATURE_COLS].astype('int32').values

    ry = pd.to_numeric(df_enc['rating_year'], errors='coerce').fillna(2000).astype(int)
    rm = pd.to_numeric(df_enc['rating_month'], errors='coerce').fillna(1).astype(int)
    ym = ry*100 + rm
    cut = np.percentile(ym, 80)
    tr_idx = ym <= cut
    va_idx = ym >  cut
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    pos = y_tr.sum(); neg = len(y_tr)-pos
    cw = {0: float(pos/(pos+neg)), 1: float(neg/(pos+neg))}

    model = AutoIntModel(field_dims=field_dims, embed_dim=48,
                         att_layer_num=4, att_head_num=4, att_res=True,
                         dnn_hidden_units=[256,128,64], dnn_dropout=0.25)
    _ = model(tf.convert_to_tensor(X_tr[:1]))

    steps_per_epoch = max(1, len(X_tr)//2048)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-3, 
                                                                    first_decay_steps=steps_per_epoch*3,
                                                                    t_mul=2.0, m_mul=0.8, alpha=1e-5)
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss=focal_loss(gamma=2.0, alpha=0.25), 
                  metrics=[tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.BinaryAccuracy(name='acc')])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True)]

    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=50, batch_size=2048,
                     class_weight=cw, callbacks=callbacks, verbose=1)

    y_pred = model(X_va, training=False).numpy().reshape(-1)
    auc   = roc_auc_score(y_va, y_pred) if len(np.unique(y_va))>1 else np.nan

    # Sample@K metrics with val set
    p10   = precision_at_k(y_pred, y_va, k=10)
    r10   = recall_at_k(y_pred, y_va, k=10)
    ndcg10= ndcg_at_k(y_pred, y_va, k=10)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({'val_auc': float(auc) if auc==auc else None,
               'val_precision_at_10': float(p10) if p10==p10 else None,
               'val_recall_at_10': float(r10) if r10==r10 else None,
               'val_ndcg_at_10': float(ndcg10) if ndcg10==ndcg10 else None,
               'history': {k: (float(v[-1]) if isinstance(v, list) else v) for k, v in hist.history.items()}
              }, open(MODEL_DIR/'metrics.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

    model.save_weights(MODEL_DIR/'autoInt_model.h5')
    model.save_weights(MODEL_DIR/'autoInt_model.weights.h5')
    print('Saved:', (MODEL_DIR/'autoInt_model.h5').as_posix(), 'and', (MODEL_DIR/'autoInt_model.weights.h5').as_posix())
